#!/usr/bin/env python3
import argparse

import cv2
import numpy as np

import py_rust_faces as prf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        "-c",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--detector",
        "-d",
        choices=["blazeface320", "blazeface640", "mtcnn"],
        default="mtcnn",
    )
    parser.add_argument(
        "--provider", "-p", choices=["cpu", "cuda", "vino"], default="cpu"
    )
    args = parser.parse_args()

    infer_provider = prf.InferProvider.OrtCpu
    if args.provider == "cuda":
        infer_provider = prf.InferProvider.OrtCuda
    elif args.provider == "vino":
        infer_provider = prf.InferProvider.OrtVino

    if args.detector == "blazeface320":
        detector = prf.blazeface(prf.BlazeFace.Net320, infer_provider=infer_provider)
    elif args.detector == "blazeface640":
        detector = prf.blazeface(prf.BlazeFace.Net640, infer_provider=infer_provider)
    else:
        detector = prf.mtcnn(infer_provider=infer_provider)

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
