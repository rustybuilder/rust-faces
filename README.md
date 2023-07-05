# Rust Faces - Face Detection Models with Rust Interface

This project aims to provide a Rust interface for multiple state-of-the-art face detection models. 

## Features

* Integration of multiple face detection models;
* Rust interface: The core implementation is written in Rust, leveraging its safety, performance, and concurrency features for preprocessing, non-maxima suppression, and other non-neural-network computationally intensive tasks.
* ONNX Runtime inference provided by [Ort](https://github.com/pykeio/ort).
* Language bindings: The project will provide bindings to Python (✔️), C (⚙️), C++ (⚙️), Java (⚙️), and C# (⚙️), enabling more developers with face technologies.
* Easy integration

## Supported Face Detection Models

The project aims to include a selection of popular and high-performing face detection models, such as:

* [x] [BlazeFace](https://github.com/zineos/blazeface) - BlazeFace640 and BlazeFace320
* [ ] MTCNN (Multi-Task Cascaded Convolutional Networks)
* [ ] EfficientDet

**Please note that the availability of specific models may vary depending on the licensing terms and open-source availability of the respective models.**

## Usage

Linux Requirements:

* libssl-dev
* pkg-config

Ubuntu: `$sudo apt install libssl-dev pkg-config`

Install the crate:

```shell
$ cargo add rust-faces --features viz
```

```rust
use rust_faces::{
    viz, DetectionParams, FaceDetection, FaceDetectorBuilder, ToArray3,
    ToRgb8,
};

pub fn main() {
    let face_detector = FaceDetectorBuilder::new(FaceDetection::BlazeFace640)
        .download()
        .build()
        .expect("Fail to load the face detector.");

    let image = image::open("tests/data/images/faces.jpg")
        .expect("Can't open test image.")
        .into_rgb8()
        .into_array3();
    let faces = face_detector.detect(image.view().into_dyn()).unwrap();

    let mut image = image.to_rgb8();
    viz::draw_faces(&mut image, faces);
    std::fs::create_dir_all("tests/output").expect("Can't create test output dir.");
    image
        .save("tests/output/should_have_smooth_design.jpg")
        .expect("Can't save test image.");
}
```

### Make OnnxRuntime shared library available

[Linux] If necessary, export your library path to the onnx runtime directory.

```shell
$ export LD_LIBRARY_PATH=<path to onnx runtime lib directory>:$LD_LIBRARY_PATH
```

If you still receive the following error message:

> PanicException: ort 1.14 is not compatible with the ONNX Runtime binary found at `onnxruntime.dll`; expected GetVersionString to return '1.14.x', but got '1.10.0'

Try to direct set the environment variable `ORT_DYLIB_PATH`: 

```bash
# bash
$ export ORT_DYLIB_PATH="<your onnx runtime dir>/onnxruntime.so"
```

```powershell
# Powershell
> $env:ORT_DYLIB_PATH="<your onnx runtime dir>/onnxruntime.dll"
```

More details on the [Ort](https://github.com/pykeio/ort) project.

## Python usage

**Requirements**

* [Rust](https://www.rust-lang.org/learn/get-started)
* [Onnx Runtime](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1) - Download one of the releases


```shell
$ pip install -e "git+https://github.com/rustybuilder/rust-faces.git#egg=py-rust-faces&subdirectory=python"
```

Usage:

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import py_rust_faces as prf

detector = prf.build_detector(prf.FaceDetection.BlazeFace640)
image = np.array(Image.open(args.image))
rects, _confidences, _landmarks = detector.detect(image)
plt.imshow(image)

for rect in rects:
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
plt.show()
```

## Contributions

Contributions to the project are welcome! If you have suggestions, bug reports, or would like to add support for additional face detection models or programming languages, please feel free to submit a pull request or open an issue.

Backlog: https://github.com/users/rustybuilder/projects/1

## License

This project is licensed under the MIT License.
