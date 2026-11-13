# Jetson Nano

Everything that runs on the Jetson (JetPack 4.6, TensorRT 8.2, Python 3.6).
The repo lives at `/mnt/usb-drive/Edge-AI-Jetson`. Engines and the C++ build are
outside the repo (`/mnt/usb-drive/models`, `/mnt/usb-drive/build/edge-ai-inference`).

| Folder | Contents |
| --- | --- |
| `cpp/` | C++ inference app (TensorRT YOLOv8-seg), see [cpp/README.md](cpp/README.md) |
| `scripts/` | Build and run: `clean_and_build.sh`, `build_engines.sh`, `test_engine_predict.sh`, `eval_engines_coco.sh`, `run_video_demo.sh` |
| `tools/` | `record_video.py` (record a clip from the CSI camera), `camera_verification.py` |
| `profiling/` | `trtexec_profile.sh`, `tegrastats_profile.sh`, `nsight_systems_profile.sh`, `nvprof_profile.sh` |

All output goes to `results/` at the repo root: `results/engines`, `results/predict`,
`results/accuracy`, `results/video` and `results/profiling`.

## Session setup

```bash
~/internet.sh
source /mnt/usb-drive/onnx-env/bin/activate
cd /mnt/usb-drive/Edge-AI-Jetson/jetson
```

## Typical order

```bash
bash scripts/build_engines.sh                 # FP32 + FP16 engines from the ONNX
bash scripts/test_engine_predict.sh           # test images  -> results/predict/jetson_*
bash scripts/eval_engines_coco.sh             # COCO subset  -> results/accuracy/jetson_*
bash scripts/run_video_demo.sh                # demo video   -> results/video
cd profiling && ./trtexec_profile.sh          # and the other profiling scripts
```

## Live stream receiver (on Windows)

```
gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false
```
