# C++ inference app (Jetson Nano)

The pipeline goes from the CSI camera to TensorRT YOLOv8-seg, then to an RTP/H.264 stream that the Windows PC displays.
The Jetson runs headless, so nothing is shown on the Jetson itself.

## Layout

| File | Responsibility |
| --- | --- |
| `src/main.cpp` | Main loop, Ctrl+C handling, camera recovery |
| `config.hpp/.cpp` | Defaults and command-line options |
| `trt_engine.hpp/.cpp` | Loads the engine, binds buffers, runs inference |
| `preprocess.hpp/.cpp` | Letterbox, BGR to RGB, HWC to CHW, written directly into the engine input |
| `yolo_seg.hpp/.cpp` | Decoding, class-aware NMS, segmentation masks |
| `visualizer.hpp/.cpp` | Mask overlay, boxes, COCO labels, status overlay |
| `video_io.hpp/.cpp` | `CsiCamera` (nvarguscamerasrc) and `RtpStreamer` (nvv4l2h264enc to udpsink) |
| `profiler.hpp/.cpp` | Per-stage latency (capture, pre, trt, post, draw, stream) |
| `cuda_utils.hpp`, `types.hpp` | RAII CUDA buffers and stream, shared structs |

## Build (on the Jetson)

```bash
mkdir -p /mnt/usb-drive/build/edge-ai-inference && cd /mnt/usb-drive/build/edge-ai-inference
cmake /mnt/usb-drive/Edge-AI-Jetson/jetson/cpp
cmake --build . -- -j4
```

## Run

Start the receiver on Windows first:

```
gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false
```

Then start the app on the Jetson:

```bash
./edge_ai_inference                      # defaults: 1280x960, masks on, stream to 192.168.55.100:5000
./edge_ai_inference --no-masks           # boxes only (faster)
./edge_ai_inference --width 1640 --height 1232
./edge_ai_inference --no-stream --max-frames 300 --stats 50   # benchmark run
./edge_ai_inference --help
```

Stop it with **Ctrl+C**, which releases the camera cleanly. If Argus ever gets stuck after a crash, run:

```bash
sudo systemctl restart nvargus-daemon
```
