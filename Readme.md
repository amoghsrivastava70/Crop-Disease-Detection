
# Crop Disease Detection Model 🌱🔍

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange.svg)](https://tensorflow.org)

A deep learning model to detect crop diseases from plant leaf images, built for Kalphathon 2025. Achieves *98% accuracy* on [Dataset Name].

---

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Installation](#installation)



## Problem Statement
Farmers often struggle to identify crop diseases early, leading to significant yield loss. This model automates disease detection using plant leaf images, helping farmers take timely action. The goal is to classify *X disease classes* across *Y crop types* with high accuracy. Allowing farmers to detect the disease in a timely manner and take steps to reduce or conttain losses.

---

## Dataset
- *Source*: [New Plant disease dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data) / [Kaggle](https://www.kaggle.com/datasets/...) 
- *Details*:
  - Total images: 22,000
  - Classes: 12 crop diseases + healthy leaves
  - Split: 73% train, 27%validation
- *Preprocessing*:
  - Resize to 226x226 pixels
  - Normalization (Batch)
  - Augmentation (rotation, flipping, brightness adjustment)

---

## Model Architecture

We use a convolutional neural network (CNN) based on a VGG-like structure with batch normalization and a classification head to help us in image clssification and disease detection.

## Architecture Overview

### Convolutional Base (Backbone)
- *Input*: (226, 226, 3)
- *Block 1*:
  - Conv2D (64 filters, 3x3) → ReLU
  - Conv2D (64 filters, 3x3) → ReLU
  - MaxPooling2D (2x2)
- *Block 2*:
  - Conv2D (128 filters, 3x3) → ReLU
  - Conv2D (128 filters, 3x3) → ReLU
  - MaxPooling2D (2x2)
- *Block 3*:
  - Conv2D (256 filters, 3x3) → ReLU
  - Conv2D (256 filters, 3x3) → ReLU
  - Conv2D (256 filters, 3x3) → ReLU
  - BatchNormalization → MaxPooling2D (2x2)
- *Block 4*:
  - Conv2D (512 filters, 3x3) → ReLU
  - Conv2D (512 filters, 3x3) → ReLU
  - Conv2D (512 filters, 3x3) → ReLU
  - BatchNormalization → MaxPooling2D (2x2)
- *Block 5*:
  - Conv2D (512 filters, 3x3) → ReLU
  - Conv2D (512 filters, 3x3) → ReLU
  - Conv2D (512 filters, 3x3) → ReLU
  - BatchNormalization → MaxPooling2D (2x2)

### Classification Head
- *Flatten*: Converts 3D features to 1D (25088 units)
- *Dense*: 256 units → ReLU → Dropout (prevents overfitting)
- *Dense*: 64 units → ReLU → Dropout
- *Output*: Dense layer with 12 units (for classification)

## Key Details
- *Total Parameters*: 21,159,820  
- *Trainable Parameters*: 6,442,572  
- *Non-Trainable Parameters*: 14,717,248 (frozen batch normalization layers)  
- *Input Size*: 226x226x3 (RGB images)  
- *Output Size*: 12 classes  

## Model Diagram (Simplified)
Input → [Conv Blocks] → Flatten → Dense → Dropout → Output 

---

## Usage
The Model can be used by passing the image of a suspected infection on leaf of the plant and be used to detect possible disease. However the model is only trained on limited number of disease and data hence the model accuracy is higher for the the data it is trained on which are 12 disease, but the model will produce false positive for other diseases. 

---

## Results 
The model upon succesfull training is able to identify 12 plant disease based on the image of the affected area of the leaf with 98% accuracy and 7% loss.
#### Model Accuracy Graph
![Model Accuracy Graph](images/accuracy.jpg)
#### Model Loss Graph
![Model Loss Graph](images/loss.jpg)

---
