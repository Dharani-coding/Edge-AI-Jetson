#include "video_io.hpp"

#include <opencv2/imgproc.hpp>

#include <sstream>

namespace edge_ai {

// ---------------------------------------------------------------- CsiCamera

CsiCamera::CsiCamera(const CameraConfig& config) : config_(config) {}

CsiCamera::~CsiCamera() { close(); }

std::string CsiCamera::pipeline() const
{
    const CameraConfig& c = config_;
    std::ostringstream p;
    p << "nvarguscamerasrc sensor-id=" << c.sensor_id;
    if (c.sensor_mode >= 0)
        p << " sensor-mode=" << c.sensor_mode;
    p << " ! video/x-raw(memory:NVMM),width=" << c.sensor_width
      << ",height=" << c.sensor_height
      << ",framerate=" << c.fps << "/1"
      << " ! nvvidconv"
      << " ! video/x-raw,format=BGRx,width=" << c.frame_width
      << ",height=" << c.frame_height
      << " ! videoconvert ! video/x-raw,format=BGR"
      << " ! appsink drop=true max-buffers=1 sync=false";
    return p.str();
}

bool CsiCamera::open()
{
    close();
    return capture_.open(pipeline(), cv::CAP_GSTREAMER);
}

void CsiCamera::close()
{
    if (capture_.isOpened())
        capture_.release();
}

bool CsiCamera::read(cv::Mat& frame)
{
    return capture_.isOpened() && capture_.read(frame) && !frame.empty();
}

// -------------------------------------------------------------- RtpStreamer

RtpStreamer::RtpStreamer(const StreamConfig& config, const cv::Size& frame_size, int fps)
    : config_(config), frame_size_(frame_size), fps_(fps)
{
}

RtpStreamer::~RtpStreamer() { close(); }

std::string RtpStreamer::pipeline() const
{
    // The leaky queue decouples the encoder from the inference loop: colour
    // conversion and encoding run on another thread, and if they fall behind
    // the oldest frame is dropped, so latency does not build up.
    std::ostringstream p;
    p << "appsrc ! video/x-raw,format=BGR"
      << " ! queue max-size-buffers=2 leaky=downstream"
      << " ! videoconvert ! video/x-raw,format=BGRx"
      << " ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
      << " ! nvv4l2h264enc insert-sps-pps=true idrinterval=" << config_.idr_interval
      << " bitrate=" << config_.bitrate << " maxperf-enable=1"
      << " ! h264parse"
      << " ! rtph264pay config-interval=1 pt=96"
      << " ! udpsink host=" << config_.host << " port=" << config_.port
      << " sync=false async=false";
    return p.str();
}

std::string RtpStreamer::receiverCommand() const
{
    std::ostringstream c;
    c << "gst-launch-1.0 udpsrc port=" << config_.port
      << " caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\""
      << " ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false";
    return c.str();
}

bool RtpStreamer::open()
{
    close();
    return writer_.open(pipeline(), cv::CAP_GSTREAMER, 0,
                        static_cast<double>(fps_), frame_size_, true);
}

void RtpStreamer::close()
{
    if (writer_.isOpened())
        writer_.release();
}

void RtpStreamer::write(const cv::Mat& frame)
{
    if (!writer_.isOpened())
        return;

    if (frame.size() == frame_size_)
    {
        writer_.write(frame);
    }
    else
    {
        cv::resize(frame, resized_, frame_size_);
        writer_.write(resized_);
    }
}

// ---------------------------------------------------------- VideoFileSource

VideoFileSource::VideoFileSource(const std::string& path) : path_(path) {}

std::string VideoFileSource::pipeline() const
{
    // No drop/leaky settings: every frame of the file must be processed.
    std::ostringstream p;
    p << "filesrc location=\"" << path_ << "\""
      << " ! qtdemux ! h264parse ! nvv4l2decoder"
      << " ! nvvidconv ! video/x-raw,format=BGRx"
      << " ! videoconvert ! video/x-raw,format=BGR"
      << " ! appsink sync=false";
    return p.str();
}

bool VideoFileSource::open()
{
    hardware_ = capture_.open(pipeline(), cv::CAP_GSTREAMER);
    if (!hardware_)
        capture_.open(path_);   // not an H.264 MP4: let OpenCV pick a decoder
    return capture_.isOpened();
}

bool VideoFileSource::read(cv::Mat& frame)
{
    return capture_.isOpened() && capture_.read(frame) && !frame.empty();
}

double VideoFileSource::fps() const
{
    const double fps = capture_.get(cv::CAP_PROP_FPS);
    return fps > 0.0 && fps < 1000.0 ? fps : 0.0;
}

long VideoFileSource::frameCount() const
{
    return static_cast<long>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
}

// ---------------------------------------------------------------- Mp4Writer

Mp4Writer::Mp4Writer(const std::string& path, const cv::Size& frame_size, double fps, int bitrate)
    : path_(path), frame_size_(frame_size), fps_(fps), bitrate_(bitrate)
{
}

Mp4Writer::~Mp4Writer() { close(); }

std::string Mp4Writer::pipeline() const
{
    std::ostringstream p;
    p << "appsrc ! video/x-raw,format=BGR"
      << " ! videoconvert ! video/x-raw,format=BGRx"
      << " ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
      << " ! nvv4l2h264enc bitrate=" << bitrate_ << " maxperf-enable=1"
      << " ! h264parse ! qtmux"
      << " ! filesink location=\"" << path_ << "\"";
    return p.str();
}

bool Mp4Writer::open()
{
    close();
    return writer_.open(pipeline(), cv::CAP_GSTREAMER, 0, fps_, frame_size_, true);
}

void Mp4Writer::write(const cv::Mat& frame)
{
    if (writer_.isOpened())
        writer_.write(frame);
}

void Mp4Writer::close()
{
    // release() sends end-of-stream and waits, so qtmux writes the MP4 index.
    if (writer_.isOpened())
        writer_.release();
}

}  // namespace edge_ai
