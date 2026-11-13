#pragma once

#include <array>
#include <chrono>
#include <string>

namespace edge_ai {

enum class Stage : int
{
    Capture,
    Preprocess,
    Inference,
    Postprocess,
    Draw,
    Stream,
    Count
};

// Short stage name: "capture", "pre", "trt", "post", "draw", "stream".
const char* stageName(Stage stage);

// Per-stage latency, averaged over a reporting window.
//
//   profiler.beginFrame();
//   ...capture...     profiler.mark(Stage::Capture);
//   ...preprocess...  profiler.mark(Stage::Preprocess);
//   profiler.endFrame();
class Profiler
{
public:
    Profiler();

    void beginFrame();
    void mark(Stage stage);   // time since the previous mark goes to `stage`
    void endFrame();

    double lastMs(Stage stage) const { return last_ms_[index(stage)]; }
    double lastFrameMs() const { return last_frame_ms_; }

    int windowFrames() const { return window_frames_; }

    // One-line summary of the current window, then starts a new window.
    std::string reportAndReset();

private:
    using Clock = std::chrono::steady_clock;
    static constexpr int kStages = static_cast<int>(Stage::Count);
    static int index(Stage s) { return static_cast<int>(s); }
    static double msSince(Clock::time_point start, Clock::time_point end);

    Clock::time_point frame_start_;
    Clock::time_point last_mark_;
    Clock::time_point window_start_;

    std::array<double, kStages> last_ms_{};
    std::array<double, kStages> sum_ms_{};
    double last_frame_ms_ = 0.0;
    double sum_frame_ms_ = 0.0;
    int window_frames_ = 0;
};

}  // namespace edge_ai
