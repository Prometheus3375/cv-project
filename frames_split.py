import argparse
import glob
import os
from pathlib import Path

import cv2


def frames_from_video(video_path):
    print(video_path)

    video = cv2.VideoCapture(str(video_path))

    count = 1
    success = 1

    output = video_path.parent / os.path.basename(video_path).split(".")[0]

    output.mkdir(exist_ok = True)

    while success:
        success, image = video.read()
        if success:
            cv2.imwrite(str(output / ("%04d_img.png" % count)), image)
            count += 1

    print(f"{count - 1} frames were extracted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Split a video on frames")
    parser.add_argument("-f", "--folders", nargs = "+", help = "List of folders with videos")
    parser.add_argument("-e", "--extensions", nargs = "+", help = "List of video extensions")
    args = parser.parse_args()

    folders = [Path(p).resolve() for p in args.folders]
    extensions = args.extensions

    for folder in folders:
        os.chdir(folder)
        for ext in extensions:
            for file in glob.glob(f"*.{ext}"):
                frames_from_video(folder / file)
