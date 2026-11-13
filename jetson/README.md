# Jetson Nano

C++ TensorRT inference (YOLOv8n-seg), plus the benchmark, accuracy and profiling scripts.

**Stack:** JetPack 4.6.6, TensorRT 8.2.1, CUDA 10.2, OpenCV 4.1.1 with GStreamer, Python 3.6. The board runs headless over SSH.

## Layout

| Path | Contents |
| --- | --- |
| `cpp/` | C++ inference app ([cpp/README.md](cpp/README.md) lists all options) |
| `scripts/` | Build and run scripts |
| `tools/` | `record_video.py` (camera clip to `data/videos/`), `camera_verification.py` |
| `profiling/` | trtexec, tegrastats, Nsight Systems and nvprof scripts |

Locations on the Jetson:

| What | Where |
| --- | --- |
| Repo | `/mnt/usb-drive/Edge-AI-Jetson` |
| Engines | `/mnt/usb-drive/models/yolov8n-seg_320x416_{fp32,fp16}_trt8.2.1.engine` |
| C++ build | `/mnt/usb-drive/build/edge-ai-inference/edge_ai_inference` |
| All output | `results/` at the repo root |

## Run order

```bash
cd /mnt/usb-drive/Edge-AI-Jetson/jetson
sudo nvpmodel -m 0 && sudo jetson_clocks            # MAXN

bash scripts/build_engines.sh          # FP32 + FP16 engines from /mnt/usb-drive/models/yolov8n-seg.onnx
bash scripts/test_engine_predict.sh    # test images -> results/predict/jetson_{fp16,fp32}
bash scripts/eval_engines_coco.sh      # COCO subset -> results/accuracy/jetson_{fp16,fp32}
bash scripts/run_video_demo.sh         # demo video  -> results/video
```

- Each run script does a clean build of the C++ app first.
- `scripts/clean_and_build.sh` builds the app without running anything.
- Inputs come from `data/`: `test_images/`, `coco_val500/` and `videos/demo_video.mp4`, copied from Windows or recorded with `tools/record_video.py`.

## Profiling

Run these from `profiling/`. Every log goes to `results/profiling/`.

| Script | Measures | Time |
| --- | --- | --- |
| `./trtexec_profile.sh` | Inference latency p50/p99 and throughput: FP32/FP16, MAXN/5 W | ~6 min |
| `./tegrastats_profile.sh` | Sustained FPS, per-stage latency, power, RAM, temperatures (5 min loop per run) | ~25 min |
| `./nsight_systems_profile.sh` | Per-stage latency from NVTX ranges (median, min, max) | ~2 min |
| `./nvprof_profile.sh` | GPU time per kernel and memory copy | ~1 min |

- On the Nano, Nsight Systems records no GPU kernels; use nvprof for GPU time.
- Each script switches power modes itself and ends in MAXN.

## Live camera stream

1. Start the receiver on Windows (needs GStreamer installed):
   ```
   gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false
   ```
2. Then start the app on the Jetson. It streams to `192.168.55.100:5000`; change that with `--host` and `--port`.
   ```bash
   /mnt/usb-drive/build/edge-ai-inference/edge_ai_inference
   ```

Stop with Ctrl+C. If the camera stays stuck after a crash, run `sudo systemctl restart nvargus-daemon`.

## Notes

- **Syncing:** files are edited on Windows and uploaded with VS Code SFTP. Sync `jetson/`, `scripts/`, `data/` and `results/`, never the whole drive: `models/`, `build/` and `onnx-env/` exist only on the Jetson.
- **Python packages:** only `pip3 install -r requirements.txt`, and only to score COCO on the Jetson; that usually runs on Windows.
