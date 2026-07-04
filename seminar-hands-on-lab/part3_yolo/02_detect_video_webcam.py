"""
Part 3.2 - Real-time detection on a video file or webcam.

YOLO11 is fast enough to run live. This streams frames, runs detection on each,
and shows/saves annotated video. Great for demonstrating real-time detection in
the seminar.

Run:
    python 02_detect_video_webcam.py --source 0                 # default webcam
    python 02_detect_video_webcam.py --source clip.mp4 --save out.mp4
    python 02_detect_video_webcam.py --source clip.mp4 --classes 0   # people only

Press 'q' in the display window to quit.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="'0' for webcam, or a path/URL to a video")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--classes", type=int, nargs="*",
                    help="restrict to COCO class ids, e.g. 0=person 2=car")
    ap.add_argument("--save", help="optional path to save annotated video")
    ap.add_argument("--no-show", action="store_true", help="don't open a window (headless)")
    args = ap.parse_args()

    from ultralytics import YOLO

    # "0" -> integer webcam index; otherwise treat as a file/URL
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)
    # stream=True yields results per frame without buffering the whole video
    stream = model.predict(
        source=source, conf=args.conf, classes=args.classes,
        stream=True, show=not args.no_show, save=bool(args.save),
    )

    frames = 0
    for _ in stream:            # iterate to actually process the stream
        frames += 1
    print(f"Processed {frames} frame(s).")
    if args.save:
        print("Annotated video saved under the Ultralytics 'runs/' folder "
              "(path printed above).")
    print("Tip: --classes 0 shows only people; add 2 for cars, 16 for dogs, etc.")


if __name__ == "__main__":
    main()
