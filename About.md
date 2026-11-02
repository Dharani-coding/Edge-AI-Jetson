Edge AI project:

Edge AI Object Detection on NVIDIA Jetson Nano Oct 2026 – Present\
Platform: Jetson Nano (JetPack 4.6.6), YOLO, ONNX, TensorRT, CUDA, GStreamer, Python\
Goal: Deploy a quantized YOLO model on a Jetson Nano for real-time, low-latency object detection from an 8 MP camera, and\
benchmark latency, throughput, and accuracy across precision levels\
• Diagnosed a non-booting Jetson Nano eMMC module over ADB (recovery kernel, blank root partition) and re-flashed JetPack\
4.6.6 via NVIDIA SDK Manager in Force Recovery mode\
• Set up headless development over USB networking and SSH, and installed required dependencies (OpenCV, PyTorch, ONNX,\
TensorRT, GStreamer) for model deployment and inference\
• Brought up live video streaming from an 8 MP camera to a laptop; reducing end-to-end latency with a hardware-accelerated\
GStreamer (H.264) pipeline (in progress)\
• Planned: export YOLO to ONNX, build TensorRT FP16 and INT8 engines, and benchmark latency, FPS, memory, and mAP\
before and after quantization against PyTorch and ONNX Runtime baselines



### Table 1: What the project should demonstrate

| **Priority**          | **What you should demonstrate** | **Evidence a recruiter can inspect**                                                                                                                                   |
| --------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Essential             | Model deployment                | Pretrained or fine-tuned YOLO exported to ONNX on the laptop, then a TensorRT engine built on the Nano with `trtexec`. Engines are rebuilt by a script, not by hand.   |
| Essential             | Accuracy validation             | mAP@50, mAP@50–95, precision and recall on the same validation set for the PyTorch/ONNX baseline and every TensorRT precision, with the accuracy change shown for each |
| Essential             | Performance optimization        | FP32 vs FP16 benchmarks, plus INT8 as an experiment. Report the result whether or not it helps, and explain it.                                                        |
| Essential             | Embedded integration            | Camera, preprocessing, TensorRT inference and boxes running locally on the Nano, with H.264 streaming to the laptop and handling of camera failure                     |
| Essential             | Software engineering            | Clean Python (and C++ for the inference path if time allows), logging, error handling, Git, unit tests (pytest, GTest), and a README that explains the setup           |
| Strong differentiator | Bottleneck analysis             | Per-stage latency breakdown (capture, preprocess, inference, postprocess, encode), one identified bottleneck, one fix, and before/after numbers                        |
| Strong differentiator | System-level measurement        | Memory, end-to-end latency, sustained FPS, temperature, CPU/GPU utilization, and power mode and power draw where you can measure it                                    |
| Strong differentiator | Reproducibility and automation  | One-command setup and benchmark, results saved as CSV/JSON with run metadata, and a CI workflow that runs the tests                                                    |
| Optional              | Robotics integration            | Detections sent from the Nano to a ROS 2 node on the laptop, useful mainly for robotics roles                                                                          |



### Table 2: Measurements to report

| **Measurement**                                  | **Why include it?**                                                                                                    |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| mAP@50 and mAP@50–95                             | Detection quality, compared against the original model                                                                 |
| Precision and recall                             | False detections vs missed objects                                                                                     |
| Accuracy change per precision (FP32, FP16, INT8) | Shows the quantization trade-off, not just the speedup                                                                 |
| Model-only latency (p50 and p99)                 | Neural-network execution time. Percentiles show spikes that averages hide.                                             |
| Per-stage latency breakdown                      | Shows where the time goes, and is the basis for the bottleneck story                                                   |
| End-to-end latency                               | Camera input to usable detection output, which is what a user experiences                                              |
| Sustained FPS over several minutes               | Real-time throughput after warm-up, not a short peak                                                                   |
| Peak RAM usage                                   | Feasibility within the Nano’s shared 4 GB                                                                              |
| Power mode and power draw                        | 5W vs MAXN, and energy per inference if you can measure it                                                             |
| Temperature, clocks and utilization              | Evidence of throttling and sustained behavior                                                                          |
| Run conditions                                   | JetPack and TensorRT versions, input size, thresholds, power mode and git commit, so anyone can reproduce your numbers |





**Done so far&#xA0;**

- Nano flashed with JetPack 4.6.6 and reachable over SSH
- CUDA 10.2, cuDNN 8.2 and TensorRT 8.2 installed, with `nvcc`, `trtexec` and the TensorRT Python module checked
- `nvarguscamerasrc` and `nvv4l2h264enc` confirmed available

**Phase 0: Finish the Nano setup (about 1 hour)**

- Free disk space (`sudo apt-get clean`, remove the desktop GUI if installed, check `df -h /`). You had only 2.7 GB free.
- Install OpenCV (`libopencv-python`) after checking its size, then `pycuda`
- Add the CUDA paths to `~/.bashrc`
- Mount the SD card through the USB reader, with `nofail` in `/etc/fstab`
- Set the power mode (`nvpmodel`, `jetson_clocks`) and note which one you use

**Phase 1: Repo setup (about 1 hour)**

- Create the GitHub repo and folder layout, with `.gitignore` for models, engines and videos
- Set up SFTPresso (or `scp`/`rsync`) and an SSH key for the Nano

**Phase 2: Baseline on the laptop (2–3 hours)**

- Pick a pretrained YOLOv8n or YOLO11n. Training isn’t needed.
- Evaluate it on COCO val (or a fixed subset) and save mAP@50, mAP@50–95, precision and recall
- Export to ONNX (try a lower opset if TensorRT 8.2 rejects it) and copy it to the Nano

**Phase 3: Engines on the Nano (3–5 hours)**

- Build FP32 and FP16 engines with `trtexec`
- Try INT8 with calibration as an experiment
- Save the `trtexec` logs

**Phase 4: Camera and stream (2–4 hours)**

- Get the 8 MP CSI camera streaming at 720p or 1080p as H.264 to your laptop with low latency
- Record a fixed test clip on the Nano for repeatable speed tests (the laptop webcam isn’t used)

**Phase 5: Inference app (4–8 hours)**

- Capture, preprocess, TensorRT inference, NMS, draw boxes, stream
- Add logging, camera-failure handling and per-stage timers

**Phase 6: Benchmarks (3–5 hours)**

- Write `bench.py` with warmup, p50/p99 latency, sustained FPS, RAM, and temperature and power from `tegrastats`
- Run every engine on the same COCO val images, save the detections, and compute mAP on the laptop
- Run speed tests on the recorded clip (speed only, since it has no labels)
- Save results as CSV/JSON with run metadata (versions, power mode, input size, git commit)

**Phase 7: Bottleneck analysis (2–4 hours)**

- Break latency down by stage, fix the slowest, and record before and after numbers
- Optional: compare the camera’s sensor modes (full 8 MP, 1080p, 720p) and measure capture and resize time

**Phase 8: Code quality (3–5 hours)**

- pytest tests and a CI workflow
- Scripts to rebuild the engines and rerun the benchmarks
- Optional: C++ inference path with GTest

**Phase 9: Write-up (3–4 hours)**

- README with the results tables first, an architecture diagram and a short demo clip
- `docs/setup-jetson.md` with your flashing story
- Replace the planned resume bullet with one containing real numbers
