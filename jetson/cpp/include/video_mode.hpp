#pragma once

#include "config.hpp"

#include <atomic>

namespace edge_ai {

// Runs the detector on every frame of a video file (config.video_path) and writes
// to config.output_dir:
//   <name>_annotated.mp4            same video with masks, boxes and labels (H.264)
//   <name>_frames/frame_NNNNNN.jpg  config.save_frames annotated frames, evenly spaced
//
// With config.loop_seconds > 0 (benchmark), the video is replayed for that long
// and nothing is saved. Either way, a summary with sustained FPS and per-stage
// averages over all frames is printed at the end.
//
// Stops early (and still finalises the MP4) when `stop` becomes true, e.g. Ctrl+C.
int runOnVideo(const AppConfig& config, const std::atomic<bool>& stop);

}  // namespace edge_ai
