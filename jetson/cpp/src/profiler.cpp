#include "profiler.hpp"

#include <cstdio>

namespace edge_ai {

namespace {
constexpr const char* kStageNames[] = {"capture", "pre", "trt", "post", "draw", "stream"};
}

Profiler::Profiler() : window_start_(Clock::now()) {}

double Profiler::msSince(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void Profiler::beginFrame()
{
    frame_start_ = Clock::now();
    last_mark_ = frame_start_;
    last_ms_.fill(0.0);
}

void Profiler::mark(Stage stage)
{
    const Clock::time_point now = Clock::now();
    const double ms = msSince(last_mark_, now);
    last_ms_[index(stage)] += ms;
    sum_ms_[index(stage)] += ms;
    last_mark_ = now;
}

void Profiler::endFrame()
{
    last_frame_ms_ = msSince(frame_start_, Clock::now());
    sum_frame_ms_ += last_frame_ms_;
    ++window_frames_;
}

std::string Profiler::reportAndReset()
{
    const Clock::time_point now = Clock::now();
    const double wall_s = msSince(window_start_, now) / 1000.0;
    const int n = window_frames_ > 0 ? window_frames_ : 1;

    char buffer[256];
    int len = std::snprintf(buffer, sizeof(buffer), "[stats] %5.1f FPS |",
                            wall_s > 0.0 ? window_frames_ / wall_s : 0.0);

    for (int s = 0; s < kStages && len < static_cast<int>(sizeof(buffer)); ++s)
        len += std::snprintf(buffer + len, sizeof(buffer) - len, " %s %.1f",
                             kStageNames[s], sum_ms_[s] / n);

    if (len < static_cast<int>(sizeof(buffer)))
        std::snprintf(buffer + len, sizeof(buffer) - len, " | total %.1f ms", sum_frame_ms_ / n);

    sum_ms_.fill(0.0);
    sum_frame_ms_ = 0.0;
    window_frames_ = 0;
    window_start_ = now;

    return buffer;
}

}  // namespace edge_ai
