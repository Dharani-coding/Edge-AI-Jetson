#!/usr/bin/env python3
"""Record a fixed-length clip from the CSI camera on the Jetson Nano (Python 3.6+).

Asks "Start recording? [y/N]", records for --duration seconds (default 30) and
saves an H.264 MP4. Encoding runs on the hardware encoder (nvv4l2h264enc), so
the CPU stays free and frames are not dropped.

    python3 record_video.py
    python3 record_video.py --duration 10 --width 1280 --height 720 --fps 60 --sensor-mode 4

The clip stops by itself after exactly duration x fps frames, so the MP4 is
always finalized. Ctrl+C stops early and still saves a playable file.
"""

import argparse
import datetime
import os
import shutil
import signal
import subprocess
import sys
import time

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # .../Edge-AI-Jetson
DEFAULT_DIR = os.path.join(REPO_DIR, "data", "videos")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=int, default=30, help="seconds to record (default 30)")
    p.add_argument("--sensor-id", type=int, default=0)
    p.add_argument("--sensor-mode", type=int, default=3, help="IMX219 mode 3 = 1640x1232 @ 30 FPS")
    p.add_argument("--width", type=int, default=1640)
    p.add_argument("--height", type=int, default=1232)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--bitrate", type=int, default=8000000, help="H.264 bitrate in bit/s")
    p.add_argument("--output", help="output .mp4 (default: data/videos/clip_<date>_<time>.mp4)")
    return p.parse_args()


def build_pipeline(args, output):
    frames = args.duration * args.fps
    return [
        "gst-launch-1.0", "-e",
        "nvarguscamerasrc", "sensor-id={}".format(args.sensor_id),
        "sensor-mode={}".format(args.sensor_mode), "num-buffers={}".format(frames), "!",
        "video/x-raw(memory:NVMM),width={},height={},framerate={}/1".format(args.width, args.height, args.fps), "!",
        "nvv4l2h264enc", "bitrate={}".format(args.bitrate), "insert-sps-pps=true", "maxperf-enable=1", "!",
        "h264parse", "!",
        "qtmux", "!",
        "filesink", "location={}".format(output),
    ]


def main():
    args = parse_args()
    if shutil.which("gst-launch-1.0") is None:
        sys.exit("gst-launch-1.0 not found (install gstreamer1.0-tools)")

    output = args.output or os.path.join(
        DEFAULT_DIR, "clip_{}.mp4".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    print("Camera {}: {}x{} @ {} FPS, {} s -> {}".format(
        args.sensor_id, args.width, args.height, args.fps, args.duration, output))
    if input("Start recording? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return 0

    log = open(output + ".log", "w")
    process = subprocess.Popen(build_pipeline(args, output), stdout=log, stderr=subprocess.STDOUT)
    start = time.time()
    print("Recording... (Ctrl+C to stop early)")

    try:
        while process.poll() is None:
            elapsed = int(time.time() - start)
            sys.stdout.write("\r  {:2d} / {} s".format(min(elapsed, args.duration), args.duration))
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        # gst-launch -e turns SIGINT into end-of-stream, so the MP4 is still finalized.
        process.send_signal(signal.SIGINT)
        print("\nStopping early...")
    finally:
        process.wait()
        log.close()
    print()

    size_mb = os.path.getsize(output) / 1e6 if os.path.exists(output) else 0.0
    if process.returncode != 0 or size_mb == 0:
        print("Recording FAILED (exit code {}). GStreamer log: {}.log".format(process.returncode, output))
        print("If the camera is busy, stop the inference app; if Argus is stuck: "
              "sudo systemctl restart nvargus-daemon")
        return 1

    os.remove(output + ".log")
    print("Saved {} ({:.1f} MB, {:.1f} s)".format(output, size_mb, time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
