# AI-BASED DRY EYE DISEASE DETECTION

This project implements AI-driven methods for:

1. Dry Eye Disease prediction from survey/questionnaire data

2. Eye Disease detection from medical eye images

3. Eye Blink detection & classification from blink image dataset

It combines Machine Learning (MLP) and Deep Learning (MobileNet, VGG-19) with Streamlit for interactive deployment.

# Features

1. Dry Eye Prediction

   Input: Dataset.xlsx survey data

   Preprocessing: Missing value handling, label encoding, train-test split

   Model: Multi-Layer Perceptron (MLP)

   Output: Stage classification (Mild / Moderate / Severe / Normal)

2. Eye Disease Prediction

   Input: Image (jpg/png)

   Preprocessing: Resize, grayscale, GLCM feature extraction

   Dataset split: Dataset/Affected/ vs Dataset/Not/

   Model: MobileNet CNN

   Output: Prediction (Affected / Not)

3. Eye Blink Detection

   Input: Image (jpg/png)

   Dataset classes: Closed, Forward Look, Left Look

   Model: VGG-19 CNN

   Output: Blink state classification

4. Streamlit UI with custom background & navigation
5. Chatbot for solving real-time queries on Dry eye disease
