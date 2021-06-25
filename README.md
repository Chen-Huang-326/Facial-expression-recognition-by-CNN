# Facial-expression-recognition-by-CNN

## Overview
This is a repo containing a facial expression recognition (FER) classifier by CNN. It aims at learning the CNN, Autoencoder and transfer learning. It contains a CNN, an Autoencoder based on CNN, a transfer learning model based on ResNet. The FER classifier aims at classifying the images into 7 different classes (7 different facial expressions), including Angry, Disgust, Fear, Happy, Neutral, Sad and Surprise. The image dataset comes from Static Facial Expressions in the Wild (SFEW), extracting the temporal image data from Acted Facial Expressions in the Wild (AFEW), which is a database comprising 675 screenshots from movies (Dhall et al., 2011). In addition, we attempt different strategies about regularizer (dropout, prunning and weight decay).

## Table of contents
1. [Overview](#overview)
2. [Author](#author)
3. [Paper](#paper)
4. [CNN](#cnn)
5. [Autoencoder](#autoencoder)
6. [Fine-tuning ResNet](#fine-tuning-resnet)
7. [Reference](#reference)

## Author
Chen Huang, hcrt520@gmail.com

## Paper
TBD

## CNN
**Version 1:** The structure of the basic CNN can be seen in the figure below. It contains 3 CNN layers and each CNN layer follows by a maxpooling layer.

**Version 2:** The structure of the concatenated CNN can be seen in the figure below. It concatenate a CNN and a simple full-connected neural network. It aims to use both the raw images and the processed data (extracted features by PHOG & LPQ) to improve the classification accuracy.

## Autoencoder
The Autoencoder is implemented by a symmetric CNN neural network, both the encoder and decoder consist of 3 CNN layers. The structure can be seen in the figure below.

## Fine-tuning ResNet
The fine-tuning ResNet aims at learning the transfer learning. It compares the performances of the fine-tuning models based on different ResNet. We use the ResNet 18/34/50/101/152 as the pretrained model separately. In our experiments, we apply two fine-tuning approaches, including last layer fine-tuning and all layer fine-tuning.

## Reference
Dhall, A., Goecke, R., Lucey, S., & Gedeon, T. (2011, November). Static facial expressions in tough conditions: Data, evaluation protocol and benchmark. In 1st IEEE International Workshop on Benchmarking Facial Image Analysis Technologies BeFIT, ICCV2011.