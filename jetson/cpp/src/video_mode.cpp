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
#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace edge_ai {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kOutputBitrate = 8000000;   // 8 Mbit/s keeps 1640x1232 detail
constexpr double kFallbackFps = 30.0;
constexpr int kStages = static_cast<int>(Stage::Count);

double secondsSince(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

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
    // Looping = benchmark: replay the video for loop_seconds and save nothing.
    const bool looping = config.loop_seconds > 0;

    VideoFileSource source(config.video_path);
    if (!source.open())
        throw std::runtime_error("Cannot open video: " + config.video_path);

    const double fps = source.fps() > 0.0 ? source.fps() : kFallbackFps;
    long total_frames = source.frameCount();   // per pass; <= 0 if unknown
    if (config.max_frames > 0)
        total_frames = total_frames > 0 ? std::min(total_frames, config.max_frames) : config.max_frames;

    const std::string name = stem(fileName(config.video_path));
    const std::string video_out = config.output_dir + "/" + name + "_annotated.mp4";
    const std::string frames_dir = config.output_dir + "/" + name + "_frames";
    const long interval = saveInterval(total_frames, config.save_frames, fps);

    std::cout << "\nVideo: " << config.video_path
              << (source.hardwareDecoding() ? " (hardware decoder)" : " (OpenCV decoder)")
              << ", " << fps << " FPS" << std::endl;
    if (looping)
    {
        std::cout << "Benchmark: looping for " << config.loop_seconds << " s, no output saved" << std::endl;
    }
    else
    {
        makeDirectories(config.output_dir);
        std::cout << "Output: " << video_out << std::endl;
        if (config.save_frames > 0)
        {
            makeDirectories(frames_dir);
            std::cout << "Saving " << config.save_frames << " frames (every " << interval
                      << ") to " << frames_dir << std::endl;
        }
    }

    TrtEngine engine(config.detector.engine_path);
    const SegModelInfo& model = engine.info();
    Letterbox letterbox(model.input_width, model.input_height);
    YoloSegDecoder decoder(model, config.detector);
    Visualizer visualizer(model.num_classes, config.detector.draw_masks);
    engine.infer();   // warm-up

    std::unique_ptr<Mp4Writer> writer;   // single pass only; created on the first frame
    Profiler profiler;
    std::array<double, kStages> stage_sum_ms{};
    cv::Mat frame;
    long frames = 0;          // frames processed in total
    long frame_in_pass = 0;   // frame index within the current pass over the video
    int passes = 1;
    int saved = 0;
    char text[256];

    std::cout << "\nProcessing... (Ctrl+C stops early" << (looping ? ")" : " and still saves the video)")
              << std::endl;
    const auto start = Clock::now();

    while (!stop)
    {
        if (config.max_frames > 0 && frames >= config.max_frames)
            break;
        if (looping && secondsSince(start) >= config.loop_seconds)
            break;

        profiler.beginFrame();
        if (!source.read(frame))
        {
            if (!looping || frame_in_pass == 0)
                break;   // end of video (or nothing readable)
            // Rewind by reopening; the time this takes is not counted as a frame.
            if (!source.open())
                throw std::runtime_error("Cannot reopen video: " + config.video_path);
            ++passes;
            frame_in_pass = 0;
            continue;
        }
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
                      frames, detections.size(), profiler.lastMs(Stage::Inference));
        visualizer.drawHud(frame, text);
        profiler.mark(Stage::Draw);

        if (!looping)
        {
            if (!writer)
            {
                writer.reset(new Mp4Writer(video_out, frame.size(), fps, kOutputBitrate));
                if (!writer->open())
                    throw std::runtime_error("Cannot open video writer: " + writer->pipeline());
            }
            writer->write(frame);

            if (interval > 0 && saved < config.save_frames && frame_in_pass % interval == 0)
            {
                std::snprintf(text, sizeof(text), "/frame_%06ld.jpg", frame_in_pass);
                if (cv::imwrite(frames_dir + text, frame))
                    ++saved;
                else
                    std::cerr << "Cannot write " << frames_dir << text << std::endl;
            }
        }
        profiler.mark(Stage::Stream);   // encode + saved frames (0 when looping)

        profiler.endFrame();
        for (int s = 0; s < kStages; ++s)
            stage_sum_ms[s] += profiler.lastMs(static_cast<Stage>(s));
        ++frames;
        ++frame_in_pass;

        if (config.stats_interval > 0 && profiler.windowFrames() >= config.stats_interval)
        {
            std::cout << profiler.reportAndReset() << " | frame " << frames;
            if (looping)
                std::cout << " | pass " << passes << " | " << static_cast<int>(secondsSince(start))
                          << "/" << config.loop_seconds << " s";
            else if (total_frames > 0)
                std::cout << "/" << total_frames;
            std::cout << std::endl;
        }
    }

    writer.reset();   // finalises the MP4 (single pass only)
    const double seconds = secondsSince(start);

    if (frames == 0)
        throw std::runtime_error("No frames could be read from " + config.video_path);

    // Whole-run summary: sustained FPS, then mean time and share per stage over every frame.
    //   Summary: 4104 frames, 5 passes, 300.0 s, sustained 13.68 FPS
    //     stage     mean ms   share
    //     capture       2.1    2.9%
    //     ...
    //     total        72.9  100.0%
    double total_ms = 0.0;
    for (int s = 0; s < kStages; ++s)
        total_ms += stage_sum_ms[s] / frames;

    std::printf("\n%s %ld frames, %d pass%s, %.1f s, sustained %.2f FPS\n",
                stop ? "Stopped early:" : "Summary:", frames, passes, passes == 1 ? "" : "es",
                seconds, frames / seconds);
    std::printf("  %-8s %8s %7s\n", "stage", "mean ms", "share");
    for (int s = 0; s < kStages; ++s)
    {
        const double mean = stage_sum_ms[s] / frames;
        std::printf("  %-8s %8.1f %6.1f%%\n", stageName(static_cast<Stage>(s)), mean,
                    total_ms > 0.0 ? 100.0 * mean / total_ms : 0.0);
    }
    std::printf("  %-8s %8.1f %6.1f%%\n", "total", total_ms, 100.0);
    std::fflush(stdout);

    if (!looping)
    {
        std::cout << "Saved " << video_out << std::endl;
        if (config.save_frames > 0)
            std::cout << "Saved " << saved << " frames to " << frames_dir << std::endl;
    }
    return 0;
}

}  // namespace edge_ai
