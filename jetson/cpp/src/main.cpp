// Edge AI inference on Jetson Nano: CSI camera -> TensorRT YOLOv8-seg -> RTP/H.264 stream.
//
// The Jetson runs headless (SSH), so annotated frames are streamed to the
// Windows PC instead of shown with cv::imshow. On Windows, run:
//   gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false
//
// Stop with Ctrl+C. The camera is then released cleanly (no Argus errors).

#include "config.hpp"
#include "image_mode.hpp"
#include "preprocess.hpp"
#include "profiler.hpp"
#include "trt_engine.hpp"
#include "video_io.hpp"
#include "visualizer.hpp"
#include "yolo_seg.hpp"

#include <signal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace edge_ai;

namespace {

std::atomic<bool> g_stop_requested{false};

// The first Ctrl+C asks for a clean shutdown. A second one forces exit,
// in case something is stuck.
void onSignal(int)
{
    if (g_stop_requested.exchange(true))
        std::_Exit(130);
}

void installSignalHandlers()
{
    struct sigaction action {};
    action.sa_handler = onSignal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);   // SSH session dropped
}

constexpr int kMaxConsecutiveReadFailures = 30;
constexpr int kMaxReopenAttempts = 5;

void sleepUnlessStopped(std::chrono::milliseconds duration)
{
    const auto end = std::chrono::steady_clock::now() + duration;
    while (!g_stop_requested && std::chrono::steady_clock::now() < end)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

// Recovers from a stalled or disconnected camera. Returns false when giving up.
bool reopenCamera(CsiCamera& camera)
{
    for (int attempt = 1; attempt <= kMaxReopenAttempts && !g_stop_requested; ++attempt)
    {
        std::cerr << "Reopening camera (attempt " << attempt << "/" << kMaxReopenAttempts << ")..." << std::endl;
        camera.close();
        sleepUnlessStopped(std::chrono::seconds(1));
        if (camera.open())
            return true;
    }

    std::cerr << "Camera could not be reopened. If Argus is stuck, run:\n"
              << "  sudo systemctl restart nvargus-daemon" << std::endl;
    return false;
}

int run(const AppConfig& config)
{
    TrtEngine engine(config.detector.engine_path);
    const SegModelInfo& model = engine.info();

    Letterbox letterbox(model.input_width, model.input_height);
    YoloSegDecoder decoder(model, config.detector);
    Visualizer visualizer(model.num_classes, config.detector.draw_masks);

    CsiCamera camera(config.camera);
    std::cout << "\nCamera pipeline:\n  " << camera.pipeline() << std::endl;
    if (!camera.open())
        throw std::runtime_error(
            "Failed to open CSI camera. Check the cable, and try: "
            "sudo systemctl restart nvargus-daemon");

    // Created after the first frame so it matches the real frame size.
    std::unique_ptr<RtpStreamer> streamer;

    Profiler profiler;
    cv::Mat frame;
    double smoothed_fps = 0.0;
    long frame_count = 0;
    int read_failures = 0;
    char hud[128];

    std::cout << "\nRunning. Press Ctrl+C to stop." << std::endl;

    while (!g_stop_requested)
    {
        profiler.beginFrame();

        if (!camera.read(frame))
        {
            if (g_stop_requested)
                break;
            if (++read_failures < kMaxConsecutiveReadFailures)
                continue;

            std::cerr << "Camera stopped delivering frames." << std::endl;
            if (!reopenCamera(camera))
                break;
            read_failures = 0;
            continue;
        }
        read_failures = 0;
        profiler.mark(Stage::Capture);

        const LetterboxInfo lb = letterbox.apply(frame, engine.input());
        profiler.mark(Stage::Preprocess);

        engine.infer();
        profiler.mark(Stage::Inference);

        const std::vector<Detection>& detections =
            decoder.process(engine.detections(), engine.prototypes(), lb, frame.size());
        profiler.mark(Stage::Postprocess);

        visualizer.drawDetections(frame, detections);
        std::snprintf(hud, sizeof(hud), "FPS %.1f | objects %zu | TRT %.1f ms",
                      smoothed_fps, detections.size(), profiler.lastMs(Stage::Inference));
        visualizer.drawHud(frame, hud);
        profiler.mark(Stage::Draw);

        if (config.stream.enabled)
        {
            if (!streamer)
            {
                streamer = std::make_unique<RtpStreamer>(config.stream, frame.size(), config.camera.fps);
                std::cout << "\nStream pipeline:\n  " << streamer->pipeline() << std::endl;
                if (!streamer->open())
                    throw std::runtime_error("Failed to open the RTP stream pipeline");
                std::cout << "\nStreaming " << frame.cols << "x" << frame.rows << " to "
                          << config.stream.host << ":" << config.stream.port
                          << ". On Windows, run:\n  " << streamer->receiverCommand() << "\n" << std::endl;
            }
            streamer->write(frame);
        }
        profiler.mark(Stage::Stream);

        profiler.endFrame();
        ++frame_count;

        const double fps = 1000.0 / std::max(1e-3, profiler.lastFrameMs());
        smoothed_fps = smoothed_fps > 0.0 ? 0.9 * smoothed_fps + 0.1 * fps : fps;

        if (config.stats_interval > 0 && profiler.windowFrames() >= config.stats_interval)
            std::cout << profiler.reportAndReset() << " | objects " << detections.size() << std::endl;

        if (config.max_frames > 0 && frame_count >= config.max_frames)
            break;
    }

    // Release the camera first so Argus shuts down cleanly, then flush the stream.
    std::cout << "\nStopping after " << frame_count << " frames..." << std::endl;
    camera.close();
    streamer.reset();
    std::cout << "Stopped cleanly." << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    AppConfig config;

    try
    {
        if (!parseArgs(argc, argv, config))
        {
            printUsage(argv[0]);
            return 0;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 2;
    }

    installSignalHandlers();
    printConfig(config);

    try
    {
        if (!config.eval_path.empty())
            return runEvaluation(config);
        if (!config.images_path.empty())
            return runOnImages(config);
        return run(config);
    }
    catch (const std::exception& error)
    {
        // run()'s destructors have already released the camera and the GPU.
        std::cerr << "\nERROR: " << error.what() << std::endl;
        return 1;
    }
}
