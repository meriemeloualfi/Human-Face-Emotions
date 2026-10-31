# Human Face Emotions

Facial emotion recognition web application built with Streamlit and TensorFlow.
The model classifies a face image into 5 emotions: **Angry, Fear, Happy, Sad, Surprise**.

## Features

- Upload an image (JPG, JPEG, PNG) or take a photo with the webcam
- Predicted emotion displayed with the probability of each class
- Progress bars and a bar chart of the probabilities
- Light and dark mode

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy, Pandas, Pillow

## Project Structure

```
Human-Face-Emotions/
├── .gitignore
├── README.md
├── app.py                  # Streamlit web application
└── mini_projet_IA.ipynb    # Project notebook
```

## Getting Started

```bash
git clone https://github.com/meriemeloualfi/Human-Face-Emotions.git
cd Human-Face-Emotions
pip install streamlit tensorflow numpy pandas pillow
streamlit run app.py
```

The app loads a trained model from `emotion_model.h5`, which must be in the project root.

## Usage

1. Choose the image source in the sidebar: upload an image or use the webcam.
2. The app resizes the image to 126×126 and predicts the emotion.
3. The predicted emotion and the probability of each class are displayed.

## Author

**Meriem El Oualfi**
GitHub: [@meriemeloualfi](https://github.com/meriemeloualfi)
