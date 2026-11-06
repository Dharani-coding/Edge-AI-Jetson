#pragma once

#include "config.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

namespace edge_ai {

// CSI camera via nvarguscamerasrc. Scaling and colour conversion to BGRx run on
// the VIC (nvvidconv). The camera is released in the destructor, which keeps
// the Argus daemon in a clean state.
class CsiCamera
{
public:
    explicit CsiCamera(const CameraConfig& config);
    ~CsiCamera();

    CsiCamera(const CsiCamera&) = delete;
    CsiCamera& operator=(const CsiCamera&) = delete;

    bool open();
    void close();
    bool isOpen() const { return capture_.isOpened(); }

    // Blocks until the next frame. Returns false on failure.
    bool read(cv::Mat& frame);

    std::string pipeline() const;

private:
    CameraConfig config_;
    cv::VideoCapture capture_;
};

// Streams BGR frames as RTP/H.264 over UDP using the hardware encoder.
// This replaces cv::imshow, because the Jetson runs headless.
class RtpStreamer
{
public:
    RtpStreamer(const StreamConfig& config, const cv::Size& frame_size, int fps);
    ~RtpStreamer();

    RtpStreamer(const RtpStreamer&) = delete;
    RtpStreamer& operator=(const RtpStreamer&) = delete;

    bool open();
    void close();

    void write(const cv::Mat& frame);

    std::string pipeline() const;

    // Matching gst-launch command for the receiving Windows PC.
    std::string receiverCommand() const;

private:
    StreamConfig config_;
    cv::Size frame_size_;
    int fps_;
    cv::VideoWriter writer_;
    cv::Mat resized_;
};

}  // namespace edge_ai
