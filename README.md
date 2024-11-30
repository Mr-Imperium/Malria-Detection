# Malaria Cell Detection Using Deep Learning

## Project Overview
This project develops a deep learning model using ResNet architecture to detect malaria-infected cells from microscopic images. The model classifies cell images as either "Parasitized" or "Uninfected" with high accuracy.

## Key Features
- Custom ResNet architecture for medical image classification
- Data augmentation to improve model generalization
- Streamlit web application for easy model deployment
- Comprehensive training and evaluation pipeline

## Dataset
- Source: Cell Images for Detecting Malaria
- Contains microscopic images of blood cells
- Binary classification: Parasitized vs. Uninfected

## Model Architecture
- Residual Network (ResNet) with custom stages
- Key components:
  - Residual modules
  - Batch normalization
  - Dropout for regularization
  - Softmax classification layer

## Technical Stack
- Python
- TensorFlow/Keras
- scikit-learn
- OpenCV
- Streamlit

## Installation

### Prerequisites
- Python 3.8+
- pip

### Steps
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset
4. Run training script:
```bash
python train.py
```

5. Launch Streamlit app:
```bash
streamlit run app.py
```

## Model Performance
- Accuracy: 96%
- Precision, Recall: 97%, 98%

## Deployment
- Local deployment via Streamlit
- Potential cloud deployment on platforms like Heroku or AWS

## Future Work
- Expand dataset
- Experiment with more advanced architectures
- Add multi-class detection capabilities

## License
[MIT]
