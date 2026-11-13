#include "video_mode.hpp"

#include "file_utils.hpp"
#include "preprocess.hpp"
#include "profiler.hpp"
#include "trt_engine.hpp"
#include "video_io.hpp"
#include "visualizer.hpp"
#include "yolo_seg.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace edge_ai {

namespace {

constexpr int kOutputBitrate = 8000000;   // 8 Mbit/s keeps 1640x1232 detail
constexpr double kFallbackFps = 30.0;

// Every `interval`-th frame is saved, so `count` frames are spread over the video.
// If the length is unknown, save one frame per second instead.
long saveInterval(long total_frames, int count, double fps)
{
    if (count <= 0)
        return 0;
    if (total_frames > 0)
        return std::max(1L, total_frames / count);
    return std::max(1L, static_cast<long>(fps + 0.5));
}

}  // namespace

int runOnVideo(const AppConfig& config, const std::atomic<bool>& stop)
{
    VideoFileSource source(config.video_path);
    if (!source.open())
        throw std::runtime_error("Cannot open video: " + config.video_path);

    const double fps = source.fps() > 0.0 ? source.fps() : kFallbackFps;
    long total_frames = source.frameCount();
    if (config.max_frames > 0)
        total_frames = total_frames > 0 ? std::min(total_frames, config.max_frames) : config.max_frames;

    const std::string name = stem(fileName(config.video_path));
    const std::string video_out = config.output_dir + "/" + name + "_annotated.mp4";
    const std::string frames_dir = config.output_dir + "/" + name + "_frames";
    makeDirectories(config.output_dir);
    if (config.save_frames > 0)
        makeDirectories(frames_dir);

    const long interval = saveInterval(total_frames, config.save_frames, fps);

    std::cout << "\nVideo: " << config.video_path
              << (source.hardwareDecoding() ? " (hardware decoder)" : " (OpenCV decoder)")
              << ", " << fps << " FPS, "
              << (total_frames > 0 ? std::to_string(total_frames) : std::string("unknown")) << " frames"
              << "\nOutput: " << video_out << std::endl;
    if (config.save_frames > 0)
        std::cout << "Saving " << config.save_frames << " frames (every " << interval
                  << ") to " << frames_dir << std::endl;

    TrtEngine engine(config.detector.engine_path);
    const SegModelInfo& model = engine.info();
    Letterbox letterbox(model.input_width, model.input_height);
    YoloSegDecoder decoder(model, config.detector);
    Visualizer visualizer(model.num_classes, config.detector.draw_masks);
    engine.infer();   // warm-up

    std::unique_ptr<Mp4Writer> writer;   // created on the first frame, at its real size
    Profiler profiler;
    cv::Mat frame;
    long frame_index = 0;
    int saved = 0;
    char text[160];

    std::cout << "\nProcessing... (Ctrl+C stops early and still saves the video)" << std::endl;
    const auto start = std::chrono::steady_clock::now();

    while (!stop)
    {
        if (config.max_frames > 0 && frame_index >= config.max_frames)
            break;

        profiler.beginFrame();
        if (!source.read(frame))
            break;   // end of video
        profiler.mark(Stage::Capture);

        const LetterboxInfo lb = letterbox.apply(frame, engine.input());
        profiler.mark(Stage::Preprocess);

        engine.infer();
        profiler.mark(Stage::Inference);

        const std::vector<Detection>& detections =
            decoder.process(engine.detections(), engine.prototypes(), lb, frame.size());
        profiler.mark(Stage::Postprocess);

        visualizer.drawDetections(frame, detections);
        std::snprintf(text, sizeof(text), "frame %ld | objects %zu | TRT %.1f ms",
                      frame_index, detections.size(), profiler.lastMs(Stage::Inference));
        visualizer.drawHud(frame, text);
        profiler.mark(Stage::Draw);

        if (!writer)
        {
            writer.reset(new Mp4Writer(video_out, frame.size(), fps, kOutputBitrate));
            if (!writer->open())
                throw std::runtime_error("Cannot open video writer: " + writer->pipeline());
        }
        writer->write(frame);

        if (interval > 0 && saved < config.save_frames && frame_index % interval == 0)
        {
            std::snprintf(text, sizeof(text), "/frame_%06ld.jpg", frame_index);
            if (cv::imwrite(frames_dir + text, frame))
                ++saved;
            else
                std::cerr << "Cannot write " << frames_dir << text << std::endl;
        }
        profiler.mark(Stage::Stream);   // encode + saved frames

        profiler.endFrame();
        ++frame_index;

        if (config.stats_interval > 0 && profiler.windowFrames() >= config.stats_interval)
        {
            std::cout << profiler.reportAndReset() << " | frame " << frame_index;
            if (total_frames > 0)
                std::cout << "/" << total_frames;
            std::cout << std::endl;
        }
    }

    writer.reset();   // finalises the MP4
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "\n" << (stop ? "Stopped early after " : "Processed ") << frame_index << " frames in "
              << seconds << " s (" << (seconds > 0 ? frame_index / seconds : 0.0) << " FPS)\n"
              << "Saved " << video_out << "\n";
    if (config.save_frames > 0)
        std::cout << "Saved " << saved << " frames to " << frames_dir << "\n";
    std::cout << std::flush;

    if (frame_index == 0)
        throw std::runtime_error("No frames could be read from " + config.video_path);
    return 0;
}

}  // namespace edge_ai
