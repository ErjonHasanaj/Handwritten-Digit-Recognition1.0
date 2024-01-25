## Table of Contents

  - [Handwritten Digit Recognizer](#handwritten-digit-recognizer)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Retraining with User Feedback](#retraining-with-user-feedback)
  - [Contributing](#contributing)
  - [License](#license)

## Handwritten Digit Recognizer

### Overview

Handwritten Digit Recognizer is a project built using Python and Keras with a Tkinter GUI. It utilizes a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The application offers real-time digit recognition and includes a user feedback mechanism to continuously enhance model accuracy.

### Features

- **Real-Time Digit Recognition**: Draw digits on the canvas and instantly receive predictions.
- **User Feedback System**: Users can correct incorrect predictions, which are saved for model retraining.
- **Retraining Script**: Periodically retrain the model with new user feedback to improve accuracy.

### Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ErjonHasanaj/Handwritten-Digit-Recognition1.0.git
cd handwritten-digit-recognizer
pip install -r requirements.txt

### Usage

To run the digit recognizer GUI:
```bash
python gui_digit_recognizer.py

Draw a digit on the canvas, and the model will predict the digit in real-time. If the prediction is incorrect, you can provide the correct digit, which will be used for future retraining.

### Retraining with User Feedback

Collected feedback data can be used to retrain the model, improving its accuracy over time:
```bash
python train_digit_recognizer.py

### Contributing

Contributions to the project are welcome. Please follow the standard fork-pull request workflow.

### License

This project is licensed under the Apache 2.0 License.

