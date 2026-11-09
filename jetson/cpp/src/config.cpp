#include "config.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace edge_ai {

namespace {

std::string nextValue(int& i, int argc, char** argv)
{
    if (i + 1 >= argc)
        throw std::invalid_argument(std::string("Missing value for ") + argv[i]);
    return argv[++i];
}

long toLong(const std::string& name, const std::string& text)
{
    size_t pos = 0;
    long value = 0;
    try { value = std::stol(text, &pos); }
    catch (const std::exception&) { pos = 0; }

    if (pos == 0 || pos != text.size())
        throw std::invalid_argument("Invalid integer for " + name + ": '" + text + "'");
    return value;
}

int toInt(const std::string& name, const std::string& text)
{
    return static_cast<int>(toLong(name, text));
}

float toFloat(const std::string& name, const std::string& text)
{
    size_t pos = 0;
    float value = 0.0f;
    try { value = std::stof(text, &pos); }
    catch (const std::exception&) { pos = 0; }

    if (pos == 0 || pos != text.size())
        throw std::invalid_argument("Invalid number for " + name + ": '" + text + "'");
    return value;
}

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::invalid_argument(message);
}

void validate(const AppConfig& c)
{
    require(c.camera.fps > 0, "--fps must be > 0");
    require(c.camera.sensor_width > 0 && c.camera.sensor_height > 0, "Sensor size must be > 0");
    require(c.camera.frame_width > 0 && c.camera.frame_height > 0, "Frame size must be > 0");
    require(c.camera.frame_width % 2 == 0 && c.camera.frame_height % 2 == 0,
            "--width/--height must be even (H.264 NV12 requirement)");
    require(c.detector.conf_threshold > 0.0f && c.detector.conf_threshold < 1.0f,
            "--conf must be in (0, 1)");
    require(c.detector.iou_threshold > 0.0f && c.detector.iou_threshold <= 1.0f,
            "--iou must be in (0, 1]");
    require(c.detector.max_detections > 0, "--max-det must be > 0");
    require(c.stream.port > 0 && c.stream.port < 65536, "--port must be 1-65535");
    require(c.stream.bitrate > 0, "--bitrate must be > 0");
    require(c.stats_interval >= 0, "--stats must be >= 0");
    require(c.max_frames >= 0, "--max-frames must be >= 0");
}

}  // namespace

bool parseArgs(int argc, char** argv, AppConfig& c)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")    return false;
        // Detector
        else if (arg == "--engine")            c.detector.engine_path = nextValue(i, argc, argv);
        else if (arg == "--conf")              c.detector.conf_threshold = toFloat(arg, nextValue(i, argc, argv));
        else if (arg == "--iou")               c.detector.iou_threshold = toFloat(arg, nextValue(i, argc, argv));
        else if (arg == "--max-det")           c.detector.max_detections = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--no-masks")          c.detector.draw_masks = false;
        // Camera
        else if (arg == "--sensor-id")         c.camera.sensor_id = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--sensor-mode")       c.camera.sensor_mode = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--sensor-width")      c.camera.sensor_width = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--sensor-height")     c.camera.sensor_height = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--fps")               c.camera.fps = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--width")             c.camera.frame_width = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--height")            c.camera.frame_height = toInt(arg, nextValue(i, argc, argv));
        // Stream
        else if (arg == "--host")              c.stream.host = nextValue(i, argc, argv);
        else if (arg == "--port")              c.stream.port = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--bitrate")           c.stream.bitrate = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--no-stream")         c.stream.enabled = false;
        // Run control
        else if (arg == "--stats")             c.stats_interval = toInt(arg, nextValue(i, argc, argv));
        else if (arg == "--max-frames")        c.max_frames = toLong(arg, nextValue(i, argc, argv));
        // Image mode
        else if (arg == "--images")            c.images_path = nextValue(i, argc, argv);
        else if (arg == "--output")            c.output_dir = nextValue(i, argc, argv);
        else
            throw std::invalid_argument("Unknown option: " + arg + " (see --help)");
    }

    validate(c);
    return true;
}

void printUsage(const char* program)
{
    const AppConfig d;
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Detector:\n"
        << "  --engine PATH        TensorRT engine        (" << d.detector.engine_path << ")\n"
        << "  --conf F             confidence threshold   (" << d.detector.conf_threshold << ")\n"
        << "  --iou F              NMS IoU threshold      (" << d.detector.iou_threshold << ")\n"
        << "  --max-det N          max detections         (" << d.detector.max_detections << ")\n"
        << "  --no-masks           boxes only, skip segmentation masks\n\n"
        << "Camera:\n"
        << "  --sensor-id N        CSI sensor id          (" << d.camera.sensor_id << ")\n"
        << "  --sensor-mode N      Argus sensor mode      (" << d.camera.sensor_mode << ")\n"
        << "  --sensor-width N     sensor mode width      (" << d.camera.sensor_width << ")\n"
        << "  --sensor-height N    sensor mode height     (" << d.camera.sensor_height << ")\n"
        << "  --fps N              capture frame rate     (" << d.camera.fps << ")\n"
        << "  --width N            processed frame width  (" << d.camera.frame_width << ")\n"
        << "  --height N           processed frame height (" << d.camera.frame_height << ")\n\n"
        << "Stream (RTP/H.264 over UDP):\n"
        << "  --host IP            receiver address       (" << d.stream.host << ")\n"
        << "  --port N             receiver UDP port      (" << d.stream.port << ")\n"
        << "  --bitrate N          H.264 bitrate, bit/s   (" << d.stream.bitrate << ")\n"
        << "  --no-stream          do not stream (benchmark only)\n\n"
        << "Run control:\n"
        << "  --stats N            print stage timings every N frames, 0 = off (" << d.stats_interval << ")\n"
        << "  --max-frames N       stop after N frames, 0 = until Ctrl+C\n"
        << "  -h, --help           show this help\n\n"
        << "Image mode (no camera, no stream):\n"
        << "  --images PATH        image file or folder to run on\n"
        << "  --output DIR         where to save annotated images + .txt detections (" << d.output_dir << ")\n";
}

void printConfig(const AppConfig& c)
{
    std::cout
        << "\nConfiguration:\n"
        << "  engine     : " << c.detector.engine_path << "\n"
        << "  thresholds : conf " << c.detector.conf_threshold
        << ", iou " << c.detector.iou_threshold
        << ", max " << c.detector.max_detections
        << (c.detector.draw_masks ? ", masks on" : ", masks off") << "\n";

    if (!c.images_path.empty())
    {
        std::cout << "  images     : " << c.images_path << " -> " << c.output_dir << "\n";
        return;
    }

    std::cout
        << "  camera     : sensor " << c.camera.sensor_id << " mode " << c.camera.sensor_mode
        << " (" << c.camera.sensor_width << "x" << c.camera.sensor_height << "@" << c.camera.fps
        << ") -> " << c.camera.frame_width << "x" << c.camera.frame_height << "\n"
        << "  stream     : ";
    if (c.stream.enabled)
        std::cout << c.stream.host << ":" << c.stream.port << " @ " << c.stream.bitrate / 1000 << " kbit/s\n";
    else
        std::cout << "disabled\n";
}

}  // namespace edge_ai
