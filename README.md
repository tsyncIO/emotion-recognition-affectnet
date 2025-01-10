# Emotion Recognition Using AffectNet with AlexNet

This project demonstrates emotion recognition using the **AffectNet** dataset with the **AlexNet** model implemented from scratch in PyTorch. The model is trained to recognize facial expressions from images, such as happy, sad, angry, and more.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
  - [Clone the repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training the Model](#training-the-model)
  - [Predicting on New Images](#predicting-on-new-images)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This project leverages the **AffectNet** dataset to train a custom **AlexNet** model for emotion recognition. The dataset contains over 1 million facial images, each labeled with one of the following emotions:

- **Happy**
- **Sad**
- **Surprise**
- **Fear**
- **Disgust**
- **Anger**
- **Neutral**

The goal of the project is to create a robust model that can accurately predict emotions from facial expressions in images.

## Setup

### Clone the repository

To get started, clone the repository:

```bash
git clone https://github.com/username/emotion-recognition-affectnet.git
cd emotion-recognition-affectnet
  


### Setup & Usage:
1. **Install Dependencies**: Instructions for installing required dependencies using ```pip install -r requirements.txt.
2. **Dataset**: Detailed information about the AffectNet dataset, including emotion categories.
3. **Usage**:
   - **Data Preprocessing**: ```python src/data_loader.py --input_path ./data/raw --output_path ./data/processed
   - **Training the Model**: ```python src/train.py --epochs 10 --batch_size 32 --learning_rate 0.01
   - **Predicting on New Images**: ```python src/predict.py --image_path ./path/to/your/image.jpg
4. **Results**: To be added on future commits.
5. **Conclusion**: Potential applications and next steps for the project will be developed incrementally...


















