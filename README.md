# Handwritten Digit Recognizer

## Overview
This project is a Handwritten Digit Recognizer built using Python and Keras with a Tkinter GUI. It leverages a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The application allows real-time digit recognition and includes a user feedback mechanism to continually improve model accuracy.

## Features
- **Real-Time Digit Recognition**: Draw digits on the canvas and get instant predictions.
- **User Feedback System**: Users can correct wrong predictions, which are saved for model retraining.
- **Retraining Script**: Periodically retrain the model with new user feedback to improve accuracy.

## Installation
To set up the project, clone the repository and install the required dependencies:

\```bash
git clone https://github.com/ErjonHasanaj/Handwritten-Digit-Recognition1.0.git

cd handwritten-digit-recognizer
pip install -r requirements.txt
\```

## Usage
To run the digit recognizer GUI:

\```bash
python gui_digit_recognizer.py
\```

Draw a digit on the canvas, and the model will predict the digit in real-time. If the prediction is incorrect, you can provide the correct digit, which will be used for future retraining.

To retrain the model with collected user feedback:

\```bash
python train_digit_recognizer.py
\```

## Retraining with User Feedback
Collected feedback data can be used to retrain the model. This helps in improving the model's accuracy over time:

\```bash
python retrain_with_feedback.py
\```

Ensure that the feedback images are stored in the specified feedback folder.

## Contributing
Contributions to the project are welcome. Please follow the standard fork-pull request workflow.

## License
[Apache 2.0 License](LICENSE)
"""
