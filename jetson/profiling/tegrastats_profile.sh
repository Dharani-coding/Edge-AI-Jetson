#!/usr/bin/env bash
# Sustained benchmark: the C++ app loops the demo video for 5 minutes per run
# (FP32 and FP16, in MAXN then 5 W) while tegrastats logs power, RAM and temperatures.
#
#   bash ../clean_and_build.sh      # once, if the app is not built yet
#   ./tegrastats_profile.sh         # 4 runs x (5 min + 1 min cool-down), about 25 minutes
#
# Output in profiling/results/:
#   tegrastats_profile_<date>_<time>.log      app output + one tegrastats summary line per run
#   tegrastats_<precision>_<mode>.log         raw tegrastats samples (every 500 ms)
#
# Summary per run, ignoring the first SKIP seconds (warm-up):
#   average power and GPU load, peak RAM and peak CPU/GPU temperature.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)       # .../Edge-AI-Jetson/jetson/profiling
REPO_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")  # .../Edge-AI-Jetson
RESULTS="$SCRIPT_DIR/results"
LOG="$RESULTS/tegrastats_profile_$(date +%Y%m%d_%H%M%S).log"
APP="/mnt/usb-drive/build/edge-ai-inference/edge_ai_inference"
VIDEO="$REPO_DIR/data/videos/demo_video.mp4"
FP32_ENGINE="/mnt/usb-drive/models/yolov8n-seg_320x416_fp32_trt8.2.1.engine"
FP16_ENGINE="/mnt/usb-drive/models/yolov8n-seg_320x416_fp16_trt8.2.1.engine"

RUN_SECONDS=300      # length of each run
SKIP_SECONDS=60      # warm-up left out of the summary
INTERVAL_MS=500      # tegrastats sampling period
COOLDOWN_SECONDS=60  # pause between runs so the board cools down

[[ -x $APP ]]   || { echo "App not built: $APP (run: bash ../clean_and_build.sh)"; exit 1; }
[[ -f $VIDEO ]] || { echo "Video not found: $VIDEO"; exit 1; }

mkdir -p "$RESULTS"
sudo -v                                         # ask for the sudo password once
trap 'sudo tegrastats --stop >/dev/null 2>&1 || true' EXIT
exec > >(tee "$LOG") 2>&1                       # everything below goes to the screen and the log

# Average power and GPU load, peak RAM and temperatures, after the warm-up samples.
summarize() {   # tegrastats log
    awk -v skip=$((SKIP_SECONDS * 1000 / INTERVAL_MS)) 'NR > skip {
        n++
        for (i = 1; i <= NF; i++) {
            if ($i == "POM_5V_IN") { split($(i+1), p, "/"); pw += p[1] }
            if ($i == "GR3D_FREQ") { gpu += $(i+1) + 0 }
            if ($i == "RAM")       { split($(i+1), r, "/"); if (r[1] + 0 > ram) ram = r[1] + 0 }
            if ($i ~ /^CPU@/)      { t = substr($i, 5) + 0; if (t > ct) ct = t }
            if ($i ~ /^GPU@/)      { t = substr($i, 5) + 0; if (t > gt) gt = t }
        }
    } END {
        if (n == 0) { print "tegrastats: no samples after warm-up"; exit }
        printf "tegrastats: %d samples | avg power %.2f W | avg GPU load %.0f%% | peak RAM %d MB | peak CPU %.1f C | peak GPU %.1f C\n",
               n, pw / n / 1000, gpu / n, ram, ct, gt
    }' "$1"
}

profile() {   # precision, mode, engine
    local teg="$RESULTS/tegrastats_$1_$2.log"
    echo
    echo "==================== ${1^^} / $2 ===================="
    rm -f "$teg"
    sudo tegrastats --interval "$INTERVAL_MS" --logfile "$teg" --start
    "$APP" --engine "$3" --video "$VIDEO" --loop-seconds "$RUN_SECONDS" --stats 300
    sudo tegrastats --stop
    summarize "$teg"
    sleep "$COOLDOWN_SECONDS"
}

echo "==> MAXN (10 W)"
sudo nvpmodel -m 0
sudo jetson_clocks
sudo nvpmodel -q
profile fp32 maxn "$FP32_ENGINE"
profile fp16 maxn "$FP16_ENGINE"

echo
echo "==> 5 W"
sudo nvpmodel -m 1
sudo jetson_clocks
sudo nvpmodel -q
profile fp32 5w "$FP32_ENGINE"
profile fp16 5w "$FP16_ENGINE"

echo
echo "==> Back to MAXN"
sudo nvpmodel -m 0
sudo jetson_clocks
echo
echo "==> Summary lines"
# Run header, the app's summary table (Summary: ... down to "total") and the tegrastats line.
awk '/^====/ || /^tegrastats:/ { print }
     /^(Summary|Stopped early):/ { table = 1 }
     table { print }
     table && $1 == "total" { table = 0 }' "$LOG"
echo "Log: $LOG"
