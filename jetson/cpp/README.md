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

Image mode runs on files instead of the camera and doesn't stream. It saves an annotated image plus `<name>.txt` for each input:

```bash
./edge_ai_inference --images /mnt/usb-drive/Edge-AI-Jetson/data/test_images \
                    --output /mnt/usb-drive/Edge-AI-Jetson/data/results/jetson
```

Download `data/results/jetson` to Windows, then run `windows\scripts\compare_with_jetson.bat`
to compare against ONNX on Windows.

Eval mode measures COCO accuracy. It writes `predictions.json` and `run_info.json` with the
evaluation thresholds conf 0.001, IoU 0.7 and 300 detections, the same as the Windows baseline:

```bash
./edge_ai_inference --engine /mnt/usb-drive/models/yolov8n-seg_320x416_fp16_trt8.2.1.engine \
                    --eval   /mnt/usb-drive/Edge-AI-Jetson/data/coco_val500/images \
                    --output /mnt/usb-drive/Edge-AI-Jetson/data/results/accuracy/jetson_fp16
```

Then score it with `scripts/coco_metrics.py --pred <output>/predictions.json`, and compare all runs
with `scripts/compare_metrics.py`.

Stop the camera mode with **Ctrl+C**, which releases the camera cleanly. If Argus ever gets stuck after a crash, run:

```bash
sudo systemctl restart nvargus-daemon
```
