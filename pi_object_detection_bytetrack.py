r"""
Live object detection + ByteTrack tracking on a Raspberry Pi, using either
the Pi Camera Module (via picamera2) or a USB webcam, with an on-screen HUD
showing FPS, latency, object count, active track count, and CPU temperature.

Defaults to the NCNN model format, which is optimized for ARM CPU inference
and is generally faster on the Pi than generic ONNX.

Setup:
    pip install ultralytics opencv-python
    # picamera2 is usually pre-installed on Raspberry Pi OS (Bullseye+).
    # If missing: sudo apt install -y python3-picamera2

    # Unzip the NCNN model exported from Colab:
    #   unzip yolov8n_ncnn_model.zip -d yolov8n_ncnn_model

Usage examples:
    # Pi Camera Module, NCNN model, live preview with track IDs
    python pi_object_detection_bytetrack.py --source picamera --show

    # USB webcam, saving annotated output and a CSV track log
    python pi_object_detection_bytetrack.py --source usb --device_index 0 \
        --output /home/pi/tracked_output.mp4 --save_csv /home/pi/tracks.csv

    # Headless benchmark run, no preview window
    python pi_object_detection_bytetrack.py --source picamera --max_frames 300
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi live object detection + ByteTrack tracking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n_ncnn_model",
        help="Model path: NCNN model dir (default), or a .onnx / .pt file",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["picamera", "usb"],
        default="picamera",
        help="Camera source: 'picamera' (CSI ribbon-cable module) or 'usb' (webcam)",
    )
    parser.add_argument(
        "--device_index",
        type=int,
        default=0,
        help="USB webcam device index (only used with --source usb)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config: 'bytetrack.yaml' or a path to a custom tracker yaml",
    )
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--show", action="store_true", help="Show a live preview window (needs a display)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save annotated video")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save a CSV track log")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until interrupted). Useful for benchmarking.",
    )
    return parser.parse_args()


def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def open_picamera(width, height):
    from picamera2 import Picamera2

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # let auto-exposure settle
    return picam2


def color_for_track(track_id):
    """Stable deterministic color per track ID, so each tracked object keeps
    a consistent color across frames."""
    if track_id < 0:
        return (128, 128, 128)
    b = (track_id * 67) % 256
    g = (track_id * 131) % 256
    r = (track_id * 197) % 256
    return (int(b), int(g), int(r))


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            f"If using NCNN, unzip it first:\n"
            f"  unzip yolov8n_ncnn_model.zip -d yolov8n_ncnn_model"
        )

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path), task="detect")

    picam2 = None
    cap = None
    if args.source == "picamera":
        picam2 = open_picamera(args.width, args.height)
    else:
        cap = cv2.VideoCapture(args.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open USB camera at index {args.device_index}")

    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_file = open(args.save_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame", "track_id", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"]
        )

    writer = None
    frame_idx = 0
    fps_smoothed = 0.0
    seen_track_ids = set()
    t_start = time.time()

    try:
        while True:
            t_frame_start = time.time()

            if picam2 is not None:
                frame_rgb = picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame, stopping.")
                    break

            # persist=True keeps ByteTrack's state across successive calls,
            # which is what maintains stable track IDs frame to frame.
            results = model.track(
                frame,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                persist=True,
                verbose=False,
            )
            result = results[0]

            annotated = frame.copy()
            n_objects = 0
            active_ids = set()

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    track_id = int(box.id[0]) if box.id is not None else -1
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_name = model.names[cls_id]

                    if track_id >= 0:
                        active_ids.add(track_id)
                        seen_track_ids.add(track_id)

                    color = color_for_track(track_id)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = (
                        f"ID {track_id} {class_name} {conf:.2f}"
                        if track_id >= 0
                        else f"{class_name} {conf:.2f}"
                    )
                    cv2.putText(
                        annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    )

                    if csv_writer is not None:
                        csv_writer.writerow(
                            [frame_idx, track_id, cls_id, class_name,
                             f"{conf:.4f}", x1, y1, x2, y2]
                        )
                    n_objects += 1

            latency_ms = (time.time() - t_frame_start) * 1000.0
            instant_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            fps_smoothed = (
                0.9 * fps_smoothed + 0.1 * instant_fps if frame_idx > 0 else instant_fps
            )

            cpu_temp = get_cpu_temp()
            temp_str = f"{cpu_temp:.1f} C" if cpu_temp is not None else "N/A"

            overlay_lines = [
                f"FPS: {fps_smoothed:.1f}",
                f"Latency: {latency_ms:.1f} ms",
                f"Objects: {n_objects}",
                f"Active tracks: {len(active_ids)}",
                f"Total tracks: {len(seen_track_ids)}",
                f"Temp: {temp_str}",
            ]
            for i, line in enumerate(overlay_lines):
                cv2.putText(
                    annotated, line, (10, 25 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            if args.output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, 15, (w, h))
                writer.write(annotated)

            if args.show:
                cv2.imshow("Pi Object Detection + ByteTrack", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(
                    f"Frame {frame_idx} | FPS: {fps_smoothed:.1f} | "
                    f"Objects: {n_objects} | Active tracks: {len(active_ids)} | Temp: {temp_str}"
                )

            if args.max_frames and frame_idx >= args.max_frames:
                print(f"Reached max_frames={args.max_frames}, stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if picam2 is not None:
            picam2.stop()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

        total_time = time.time() - t_start
        print(f"\nProcessed {frame_idx} frames in {total_time:.1f}s")
        if frame_idx > 0:
            print(f"Average FPS: {frame_idx / total_time:.2f}")
        print(f"Unique track IDs seen: {len(seen_track_ids)}")
        if args.output:
            print(f"Annotated video saved to: {args.output}")
        if args.save_csv:
            print(f"Track log saved to: {args.save_csv}")


if __name__ == "__main__":
    main()
