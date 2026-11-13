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

// Reads a video file. H.264 MP4s (e.g. from record_video.py) are decoded on the
// hardware decoder (nvv4l2decoder); other files fall back to OpenCV's own reader.
class VideoFileSource
{
public:
    explicit VideoFileSource(const std::string& path);

    VideoFileSource(const VideoFileSource&) = delete;
    VideoFileSource& operator=(const VideoFileSource&) = delete;

    bool open();
    bool read(cv::Mat& frame);   // false at end of file

    double fps() const;          // 0 if unknown
    long frameCount() const;     // <= 0 if unknown
    bool hardwareDecoding() const { return hardware_; }
    std::string pipeline() const;

private:
    std::string path_;
    cv::VideoCapture capture_;
    bool hardware_ = false;
};

// Writes BGR frames to an H.264 MP4 with the hardware encoder.
// No frames are dropped; close() (or the destructor) finalises the file.
class Mp4Writer
{
public:
    Mp4Writer(const std::string& path, const cv::Size& frame_size, double fps, int bitrate);
    ~Mp4Writer();

    Mp4Writer(const Mp4Writer&) = delete;
    Mp4Writer& operator=(const Mp4Writer&) = delete;

    bool open();
    void write(const cv::Mat& frame);
    void close();

    std::string pipeline() const;

private:
    std::string path_;
    cv::Size frame_size_;
    double fps_;
    int bitrate_;
    cv::VideoWriter writer_;
};

}  // namespace edge_ai
