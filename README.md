# Emotion Recognition Using AffectNet with AlexNet

This project demonstrates emotion recognition using the AffectNet dataset with the AlexNet model implemented from scratch in PyTorch. The model is trained to recognize facial expressions from images, such as happy, sad, angry, etc.

## Setup

### Clone the repository

To get started, clone the repository:

```bash
git clone https://github.com/username/emotion-recognition-affectnet.git
cd emotion-recognition-affectnet


###Install the required dependencies:

```pip install -r requirements.txt

###Data
The AffectNet dataset consists of 1M facial images labeled with one of the following emotions:

Happy
Sad
Surprise
Fear
Disgust
Anger
Neutral
Usage

###To preprocess the data:
```python src/data_loader.py --input_path ./data/raw --output_path ./data/processed

###To train the model:
```python src/train.py --epochs 10 --batch_size 32 --learning_rate 0.01

###To make predictions:
```python src/predict.py --image_path ./path/to/your/image.jpg

###Results
To be added on future commits.

###Conclusion
Potential applications and next steps for the project will be developed incrementally.
