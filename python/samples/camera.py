import argparse

import cv2
import numpy as np

import py_rust_faces as prf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    detector = prf.build_detector(
        prf.FaceDetection.BlazeFace640, infer_provider=prf.InferProvider.OrtCuda
    )

    capture = cv2.VideoCapture(args.camera)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects, _confidences, landmarks = detector.detect(frame_rgb)
        rects = rects.astype(np.int32)
        landmarks = landmarks.astype(np.int32)
        for r, landmarks in zip(rects, landmarks):
            cv2.rectangle(
                frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2
            )

            for lm in landmarks:
                cv2.circle(frame, (lm[0], lm[1]), 2, (0, 255, 0), 2)

        cv2.imshow("Faces", frame)
        key = chr(cv2.waitKey(1) & 0xFF)

        if key == "q":
            break


if __name__ == "__main__":
    main()
