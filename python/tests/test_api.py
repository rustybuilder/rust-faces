from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import py_rust_faces as prf


@pytest.fixture()
def sample_image():
    return np.array(
        Image.open(Path(__file__).absolute().parents[2] / "tests/data/images/faces.jpg")
    )


def test_blazeface(sample_image):
    detector = prf.build_detector(prf.FaceDetection.BlazeFace640)
    faces, scores, landmarks = detector.detect(sample_image)
    assert faces.shape == (5, 4)
    assert scores.shape == (5,)
    assert landmarks.shape[0] == 5
