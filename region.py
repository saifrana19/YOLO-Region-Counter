# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

current_region = None

# Global counting_regions variable
counting_regions = [
    {
        "name": "YOLO Polygon Region",
        "polygon": Polygon([(100, 100), (500, 100), (500, 400), (100, 400)]),  # Larger rectangle
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # Red
        "text_color": (255, 255, 255),  # White
    },
    {
        "name": "YOLO Circle Region", 
        "polygon": Polygon([(300, 300), (400, 200), (500, 300), (400, 400)]),  # Diamond shape
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # Cyan
        "text_color": (0, 0, 0),  # Black
    },
]

# Store original regions for reset functionality
original_regions = [
    {
        "name": "YOLO Polygon Region",
        "polygon": Polygon([(100, 100), (500, 100), (500, 400), (100, 400)]),
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),
        "text_color": (255, 255, 255),
    },
    {
        "name": "YOLO Circle Region", 
        "polygon": Polygon([(300, 300), (400, 200), (500, 300), (400, 400)]),
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),
        "text_color": (0, 0, 0),
    },
]


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """
    Handle mouse events for region manipulation in the video frame.
    """
    global current_region, counting_regions

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def reset_regions():
    """Reset counting regions to original positions"""
    global counting_regions
    counting_regions = [
        {
            "name": "YOLO Polygon Region",
            "polygon": Polygon([(100, 100), (500, 100), (500, 400), (100, 400)]),
            "counts": 0,
            "dragging": False,
            "region_color": (255, 42, 4),
            "text_color": (255, 255, 255),
        },
        {
            "name": "YOLO Circle Region", 
            "polygon": Polygon([(300, 300), (400, 200), (500, 300), (400, 400)]),
            "counts": 0,
            "dragging": False,
            "region_color": (37, 255, 225),
            "text_color": (0, 0, 0),
        },
    ]
    print("Regions reset to original positions!")


def run(
    weights: str = "yolo11n.pt",
    source: str | None = None,
    device: str = "cpu",
    view_img: bool = True,
    save_img: bool = False,
    exist_ok: bool = False,
    classes: list[int] | None = None,
    line_thickness: int = 2,
    track_thickness: int = 2,
    region_thickness: int = 2,
) -> None:
    """
    Run object detection and counting within specified regions using YOLO and ByteTrack.
    """
    global counting_regions  # Declare global to modify the global variable
    
    # Setup Model
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract classes names
    names = model.names

    # Video setup - Use webcam (source=0)
    videocapture = cv2.VideoCapture(0)  # 0 for default webcam
    
    # Set higher resolution for better quality
    videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    videocapture.set(cv2.CAP_PROP_FPS, 30)
    
    frame_width = int(videocapture.get(3))
    frame_height = int(videocapture.get(4))
    fps = int(videocapture.get(5))
    
    print(f"Camera resolution: {frame_width}x{frame_height}")
    print(f"Camera FPS: {fps}")

    # Output setup
    if save_img:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(save_dir / "webcam_output.avi"), fourcc, fps, (frame_width, frame_height))

    # Create window and set mouse callback at the start
    cv2.namedWindow("YOLO Real-Time Region Counter")
    cv2.setMouseCallback("YOLO Real-Time Region Counter", mouse_callback)
    
    print("Camera started! Showing real-time object detection...")
    print("Instructions:")
    print("- Press 'q' to quit")
    print("- Press 'p' to pause") 
    print("- Press 'r' to reset regions")
    print("- Click and drag regions to move them")
    print("- Detected objects will be counted in each region")

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            print("Failed to grab frame")
            break

        # Flip frame horizontally for mirror effect (more natural for webcam)
        frame = cv2.flip(frame, 1)

        # Extract the results with higher confidence threshold
        results = model.track(frame, persist=True, classes=classes, conf=0.5, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, f"{names[cls]} {track_id}", color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # Check if detection inside region
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coordinates = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            
            # Draw background for text
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            # Draw count text
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, region_text_color, line_thickness + 1
            )
            # Draw region boundary
            cv2.polylines(frame, [polygon_coordinates], isClosed=True, color=region_color, thickness=region_thickness + 2)

        # Display information on the frame
        cv2.putText(frame, "YOLO Real-Time Object Counter", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit | 'p' to pause | 'r' to reset", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Drag regions with mouse", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show FPS and object count
        total_objects = sum(region["counts"] for region in counting_regions)
        cv2.putText(frame, f"Total Objects: {total_objects}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("YOLO Real-Time Region Counter", frame)

        # Save frame if enabled
        if save_img:
            video_writer.write(frame)

        # Reset counts for next frame
        for region in counting_regions:
            region["counts"] = 0

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("p"):  # Pause functionality
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
        elif key == ord("r"):  # Reset regions
            reset_regions()

    # Cleanup
    videocapture.release()
    if save_img:
        video_writer.release()
        print(f"Video saved to: {save_dir}")
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")


def parse_opt() -> argparse.Namespace:
    """Parse command line arguments for the region counting application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="initial weights path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, default="0", help="webcam source (0 for default camera)")
    parser.add_argument("--view-img", action="store_true", default=True, help="show results")
    parser.add_argument("--save-img", action="store_true", default=False, help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    return parser.parse_args()


def main(options: argparse.Namespace) -> None:
    """Execute the main region counting functionality with the provided options."""
    # Force view_img to True and use webcam
    options.view_img = True
    options.source = "0"  # Force webcam
    
    print("🚀 Starting YOLO Real-Time Webcam Object Counter...")
    print("📷 Initializing camera...")
    
    run(**vars(options))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)