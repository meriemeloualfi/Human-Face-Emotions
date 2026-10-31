# Human Face Emotions

Deep learning project that recognizes human facial emotions from an image, with an interactive
web interface built with Streamlit. The model classifies a face into 5 emotions:
**Angry, Fear, Happy, Sad, Surprise**.

## Features

- Emotion prediction from an uploaded image (JPG, JPEG, PNG) or a webcam photo
- Predicted emotion with a probability score for each class
- Probability bar chart and progress bars
- Light and dark mode
- Model training and evaluation documented in a Jupyter notebook

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Deep learning | TensorFlow / Keras |
| Web app | Streamlit |
| Data and images | NumPy, Pandas, Pillow |
| Training environment | Jupyter Notebook / Google Colab |

## Project Structure

```
Human-Face-Emotions/
├── app.py                  # Streamlit web application
├── mini_projet_IA.ipynb    # Data preparation, model training and evaluation
├── emotion_model.h5        # Trained model (required to run the app)
├── requirements.txt
└── README.md
```

## Dataset

[Name of the dataset, with link, e.g. Kaggle]
- Classes: Angry, Fear, Happy, Sad, Surprise
- Number of images: [fill in]
- Train / validation / test split: [fill in]

## Model

- Input: RGB images resized to 126×126 pixels, pixel values scaled to [0, 1]
- Output: probabilities for the 5 emotion classes (softmax)
- Architecture: [CNN / transfer learning with MobileNetV2, etc.]
- Optimizer, loss, epochs: [fill in]

### Results

| Metric | Value |
|--------|-------|
| Training accuracy | [fill in] |
| Validation accuracy | [fill in] |
| Test accuracy | [fill in] |

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
git clone https://github.com/meriemeloualfi/Human-Face-Emotions.git
cd Human-Face-Emotions
pip install -r requirements.txt
```

### Run the app

Make sure `emotion_model.h5` is in the project root, then run:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

### Train the model yourself

Open `mini_projet_IA.ipynb` in Jupyter or Google Colab, run the cells in order, and save the
trained model as `emotion_model.h5` in the project root.

## Usage

1. Choose the image source in the sidebar: upload an image or take a photo with the webcam.
2. The model analyzes the face and displays the predicted emotion.
3. Check the probability of each emotion in the progress bars and the chart.

## Limitations

- The model only recognizes 5 emotions (no Neutral or Disgust).
- Results work best on clear, front-facing, well-lit faces.
- The app does not detect or crop faces: the whole image is resized and classified.

## Future Improvements

- Add face detection (OpenCV, MediaPipe) before classification
- Add more emotion classes
- Real-time video emotion recognition
- Deploy the app online (Streamlit Community Cloud, Hugging Face Spaces)

## Author

**Meriem El Oualfi**
GitHub: [@meriemeloualfi](https://github.com/meriemeloualfi)
