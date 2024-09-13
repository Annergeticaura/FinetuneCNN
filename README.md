# FinetuneCNN
Finetuned Resnet-50 on UTK dataset
# Age Detection with Fine-Tuned CNN on UTK Dataset
## Overview
This project focuses on fine-tuning a pre-trained Convolutional Neural Network (CNN), specifically ResNet50, for age detection using the UTKFace dataset. The goal is to predict a person's age from facial images, with a specific emphasis on leveraging transfer learning to enhance model accuracy and reduce training time.

## Features
Fine-tuned ResNet50 model for age detection.
Transfer Learning: Utilizes pre-trained weights from the ImageNet dataset.
Minimal data preprocessing required due to transfer learning.
High accuracy with real-world face images from the UTKFace dataset.
Metrics such as mean absolute error (MAE) are used to evaluate model performance.
# Dataset
The UTKFace dataset contains over 20,000 images of faces with annotations for age, gender, and ethnicity. For this project, only the age label is used to train the model.

## Dataset Structure:
Images of faces in various lighting conditions, angles, and resolutions.
Age annotations span from 0 to 116 years.
## Preprocessing:
All images are resized to 224x224 pixels.
Images are normalized by dividing pixel values by 255.
The dataset is split into training, validation, and test sets.
# Model Architecture
The base model used for fine-tuning is ResNet50 with pre-trained weights from ImageNet. The architecture includes:

ResNet50 base layers: The convolutional and pooling layers remain frozen to leverage the learned features.
Custom Dense layers: After the base model, a Flatten layer followed by Dense layers is added for prediction.
Output layer: A single neuron with linear activation is used for predicting age as a continuous value.
# License
This is project is under MIT License
