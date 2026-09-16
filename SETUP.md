# YOLOv8n + ByteTrack on Raspberry Pi

Live object detection and tracking on a Raspberry Pi, using an NCNN-exported YOLOv8n model (80 COCO classes) and Ultralytics' built-in ByteTrack.

## What's in this repo

```
yolov8n\_ncnn\_model/            NCNN model (ARM-optimized, exported from stock yolov8n)
pi\_object\_detection\_bytetrack.py
requirements.txt
```

## Requirements

* Raspberry Pi 4 or 5 (a Pi 5 will get noticeably better FPS)
* Raspberry Pi OS Bullseye or newer
* Either the Raspberry Pi Camera Module (CSI ribbon cable) or a USB webcam
* Python 3.9+

## Setup

Update the system and install the camera library via apt — `picamera2` depends on system-level `libcamera` bindings and will not work if installed through pip.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv
```

Clone the repo and set up a virtual environment:

```bash
git clone <this-repo-url>
cd <repo-folder>

python3 -m venv --system-site-packages venv
source venv/bin/activate
```

`--system-site-packages` is important — it lets the venv see the apt-installed `picamera2` instead of trying to reinstall it through pip, which usually fails.

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

## Running it

Pi Camera Module, live preview:

```bash
python pi\_object\_detection\_bytetrack.py --source picamera --show
```

USB webcam instead:

```bash
python pi\_object\_detection\_bytetrack.py --source usb --device\_index 0 --show
```

Headless (no display attached, e.g. over SSH) — drop `--show` and save output instead:

```bash
python pi\_object\_detection\_bytetrack.py --source picamera \\
  --output /home/pi/tracked\_output.mp4 \\
  --save\_csv /home/pi/tracks.csv
```

Quick FPS benchmark (stops after 300 frames and prints the average):

```bash
python pi\_object\_detection\_bytetrack.py --source picamera --max\_frames 300
```

## Useful flags

|Flag|Default|What it does|
|-|-|-|
|`--model`|`yolov8n\_ncnn\_model`|Path to the model directory|
|`--source`|`picamera`|`picamera` or `usb`|
|`--device\_index`|`0`|USB camera index (only used with `--source usb`)|
|`--width`, `--height`|`640`, `480`|Capture resolution|
|`--conf`|`0.25`|Detection confidence threshold|
|`--iou`|`0.45`|NMS IoU threshold|
|`--imgsz`|`640`|Inference size — lower this first if FPS is too low|
|`--tracker`|`bytetrack.yaml`|Tracker config, or a path to a custom one|
|`--show`|off|Live preview window (needs a display)|
|`--output`|none|Save annotated video to this path|
|`--save\_csv`|none|Log every detection/track to a CSV|
|`--max\_frames`|unlimited|Stop after N frames, for benchmarking|

## On-screen overlay

FPS, per-frame latency, current object count, active track count, cumulative unique tracks seen, and CPU temperature (read from `/sys/class/thermal/thermal\_zone0/temp`).

## If FPS is too low

Lower `--imgsz` before anything else — dropping to 480 or 320 has a much bigger effect than tuning the tracker. ByteTrack's association step also runs on CPU alongside inference, so tracking is inherently slower than plain detection on the same hardware.

## Tuning the tracker

Copy `bytetrack.yaml` from your Ultralytics install, edit it, and point `--tracker` at your copy:

```yaml
tracker\_type: bytetrack
track\_high\_thresh: 0.5
track\_low\_thresh: 0.1
new\_track\_thresh: 0.6
track\_buffer: 30      # frames a lost track stays alive before being dropped
match\_thresh: 0.8
```

Raise `track\_buffer` if tracks are dropping during brief occlusions. Tracking can only work with detections the model actually produces — if objects are being missed rather than losing their ID, the fix is `--conf` or the model itself, not the tracker config.

## Troubleshooting

**`picamera2` import fails inside the venv** — you likely forgot `--system-site-packages` when creating it, or forgot the apt install step. Recreate the venv with that flag.

**Camera not found** — confirm it's enabled: `sudo raspi-config` → Interface Options → Camera, then reboot. For CSI cameras, check the ribbon cable orientation.

**Model not found error on startup** — make sure `yolov8n\_ncnn\_model/` is in the same directory you're running the script from, or pass its path explicitly with `--model`.

