import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import py_rust_faces as prf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=Path(__file__).absolute().parents[2] / "tests/data/images/faces.jpg",
    )
    args = parser.parse_args()
    detector = prf.build_detector(prf.FaceDetection.BlazeFace640)
    image = np.array(Image.open(args.image))
    rects, _confidences, landmarks = detector.detect(image)

    plt.imshow(image)
    for rect, lms in zip(rects, landmarks):
        plt.gca().add_patch(
            Rectangle(
                (rect[0], rect[1]),
                rect[2],
                rect[3],
                edgecolor="red",
                facecolor="none",
                lw=2,
            )
        )
        plt.plot(lms[:, 0], lms[:, 1], "ro")
        
    plt.show()

if __name__ == "__main__":
    main()
