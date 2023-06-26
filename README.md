# Rust Faces - Face Detection Models with Rust Interface

This project aims to provide a Rust interface for multiple state-of-the-art face detection models. The interface will include bindings to popular programming languages such as Python, C, C++, Java, and C#, making it accessible to a wide range of developers.

## Features

* Integration of multiple face detection models: The project will incorporate various state-of-the-art face detection models, allowing users to choose the most suitable one for their specific requirements.
* Rust interface: The core implementation is written in Rust, leveraging its safety, performance, and concurrency features for preprocessing, non-maxima suppression, and other non-neural-network computationally intensive tasks.
* ONNX Runtime inference provided by [ort](https://github.com/pykeio/ort).
* Language bindings: The project will provide bindings to Python (W.I.P.), C (⚙️), C++ (⚙️), Java (⚙️), and C# (⚙️), enabling developers to utilize the face detection models in their preferred programming language.
* Easy integration: The interface will be designed to be simple and easy to integrate into existing projects, minimizing the development effort required.

## Supported Face Detection Models

The project aims to include a selection of popular and high-performing face detection models, such as:

* [ ] RetinaFace
* [ ] BlazeFace - W.I.P.
* [ ] MTCNN (Multi-Task Cascaded Convolutional Networks)
* [ ] EfficientDet

**Please note that the availability of specific models may vary depending on the licensing terms and open-source availability of the respective models.**

## Usage

To use the face detection models via the Rust interface, follow the steps below:

Clone the repository:

```shell
cargo add rust-faces
```

Choose your preferred language binding:

For Python, use the provided Python wrapper:

```shell
import py_rust_faces as rfaces

# Initialize the face detection model
face_det = rfaces.build_detector(rfaces.Method.BlazeFace)

# Load an image
image = np.imread("test.jpg");

# Perform face detection
faces, scores, landmarks = face_det.detect(image)
```


## Contributions

Contributions to the project are welcome! If you have suggestions, bug reports, or would like to add support for additional face detection models or programming languages, please feel free to submit a pull request or open an issue.

Before making substantial changes, please ensure to discuss them with the project maintainers to align with the project's goals and maintain compatibility with the existing codebase.

## License

This project is licensed under the MIT License.