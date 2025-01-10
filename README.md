# Emotion Recognition Using AffectNet with AlexNet

This project demonstrates emotion recognition using **AffectNet** dataset with the **AlexNet** model implemented from scratch in PyTorch. The model is trained to recognize facial expressions from images, such as happy, sad, angry, etc.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/emotion-recognition-affectnet.git
   cd emotion-recognition-affectnet

2. Install the required dependencies:
   pip install -r requirements.txt

3. Data: The AffectNet dataset consists of 1M facial images labeled with one of the following emotions: Happy, Sad, Surprise, Fear, Disgust, Anger, Neutral

4. Usage:
   a. Data Preprocessing: 
      To preprocess the data;
      python src/data_loader.py --input_path ./data/raw --output_path ./data/processed

   b. Training the Model:
      To train the model;
      python src/train.py --epochs 10 --batch_size 32 --learning_rate 0.01

   c. Predicting on New Images:
      To make predictions;
      python src/predict.py --image_path ./path/to/your/image.jpg



