import cv2
import os
import argparse
import glob
from pathlib import Path


def frames_from_video(video_path):
    video = cv2.VideoCapture(video_path)

    count = 1
    success = 1

    output = video_path.parent / os.path.basename(video_path).split(".")[0]

    while success:
        success, image = video.read()
        cv2.imwrite(output / ("%04d_img.png" % count), image)
        count += 1

    print(f"{count - 1} frames were extracted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a video on frames")
    parser.add_argument("-f", "--folders", nargs="+", help="List of folders with videos")
    parser.add_argument("-e", "--extensions", nargs="+", help="List of video extensions")
    args = parser.parse_args()

    folders = [Path(p).resolve() for p in args.folders]
    extensions = args.extensions

    for folder in folders:
        os.chdir(folder)
        for ext in extensions:
            for file in glob.glob(f"*.{ext}"):
                frames_from_video(folder / file)
