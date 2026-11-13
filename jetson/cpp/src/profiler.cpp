#include "profiler.hpp"

#include <cstdint>
#include <cstdio>

#ifdef EDGE_AI_NVTX
#include <nvToolsExt.h>
#endif

namespace edge_ai {

namespace {
constexpr const char* kStageNames[] = {"capture", "pre", "trt", "post", "draw", "stream"};

#ifdef EDGE_AI_NVTX
// One colour per stage on the Nsight Systems timeline; the last one is "frame".
constexpr uint32_t kRangeColors[] = {
    0xFF4E79A7,   // capture
    0xFFF28E2B,   // pre
    0xFF59A14F,   // trt
    0xFFE15759,   // post
    0xFFB07AA1,   // draw
    0xFF76B7B2,   // stream
    0xFF9C9C9C,   // frame
};
#endif
}  // namespace

const char* stageName(Stage stage)
{
    return kStageNames[static_cast<int>(stage)];
}

Profiler::Profiler() : window_start_(Clock::now()) {}

Profiler::~Profiler() { popAllRanges(); }

double Profiler::msSince(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ------------------------------------------------------------------ NVTX

void Profiler::pushRange(int stage_or_frame)
{
#ifdef EDGE_AI_NVTX
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = kRangeColors[stage_or_frame];
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = stage_or_frame < kStages ? kStageNames[stage_or_frame] : "frame";
    nvtxRangePushEx(&attr);
    ++open_ranges_;
#else
    (void)stage_or_frame;
#endif
}

void Profiler::popAllRanges()
{
#ifdef EDGE_AI_NVTX
    for (; open_ranges_ > 0; --open_ranges_)
        nvtxRangePop();
#endif
}

// ------------------------------------------------------------------ Timing

void Profiler::beginFrame()
{
    popAllRanges();   // a frame abandoned with `continue` (e.g. read failure)
    pushRange(kStages);                        // "frame"
    pushRange(index(Stage::Capture));          // first stage

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

#ifdef EDGE_AI_NVTX
    // Close this stage's range and open the next one.
    if (open_ranges_ > 1)
    {
        nvtxRangePop();
        --open_ranges_;
    }
    if (index(stage) + 1 < kStages)
        pushRange(index(stage) + 1);
#endif
}

void Profiler::endFrame()
{
    last_frame_ms_ = msSince(frame_start_, Clock::now());
    sum_frame_ms_ += last_frame_ms_;
    ++window_frames_;
    popAllRanges();
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
