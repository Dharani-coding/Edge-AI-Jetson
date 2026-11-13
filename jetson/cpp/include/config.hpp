#pragma once

#include <string>

namespace edge_ai {

struct CameraConfig
{
    int sensor_id = 0;
    int sensor_mode = 3;        // IMX219 mode 3: 1640x1232 @ 30 FPS
    int sensor_width = 1640;
    int sensor_height = 1232;
    int fps = 30;
    // Frame size delivered to the app. Scaling is done by nvvidconv (hardware),
    // so a smaller frame cuts CPU work for drawing, masks and encoding.
    int frame_width = 1280;
    int frame_height = 960;
};

struct StreamConfig
{
    bool enabled = true;
    std::string host = "192.168.55.100";   // Windows PC on the USB network
    int port = 5000;
    int bitrate = 4000000;
    int idr_interval = 15;
};

struct DetectorConfig
{
    std::string engine_path = "/mnt/usb-drive/models/yolov8n-seg_fp16.engine";
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_detections = 100;
    bool draw_masks = true;
};

struct AppConfig
{
    CameraConfig camera;
    StreamConfig stream;
    DetectorConfig detector;
    int stats_interval = 30;    // frames between console stats lines, 0 = off
    long max_frames = 0;        // 0 = run until Ctrl+C

    // Image mode: if set, run on these image(s) instead of the camera.
    std::string images_path;            // image file or folder
    std::string output_dir = "results"; // annotated images + <name>.txt, or predictions.json

    // Eval mode: if set, write COCO predictions.json for the images in this folder.
    // Detector defaults switch to COCO-evaluation values (see kEval* below).
    std::string eval_path;

    // Video mode: if set, annotate this video file into <output_dir>/<name>_annotated.mp4.
    std::string video_path;
    int save_frames = 0;                // also save this many annotated frames as JPG
};

// Detector defaults in --eval mode (same as Ultralytics validation and
// windows/scripts/eval_onnx_coco.py). Explicit --conf/--iou/--max-det still win.
constexpr float kEvalConfThreshold = 0.001f;
constexpr float kEvalIouThreshold = 0.7f;
constexpr int kEvalMaxDetections = 300;

// Parses command-line options into `config`.
// Returns false if --help was requested. Throws std::invalid_argument on bad input.
bool parseArgs(int argc, char** argv, AppConfig& config);

void printUsage(const char* program);
void printConfig(const AppConfig& config);

}  // namespace edge_ai
