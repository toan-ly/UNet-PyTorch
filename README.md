# UNet Image Segmentation with Carvana Dataset

This project implements a UNet-based image segmentation model which is introduced in the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) using PyTorch. The model is trained on the [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) dataset. The model is designed to segment car images by accurately identifying the car pixels.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)

## Overview
UNet is a convolutional neural network architecture designed for biomedical image segmentation. In this project, we leverage the UNet model to perform image segmentation on car images from the Carvana dataset.

<p align="center">
  <img src="./assets/unet.png" height="90%%" width="90%%"/>
</p>

## Dataset
The dataset used in this project is from the [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge). The dataset includes:
- **Train Images**: High-quality images of cars.
- **Train Masks**: Binary masks corresponding to the car regions in the train images.
- **Test Images**: High-quality images for testing the model.

## Model Architecture
The UNet architecture consists of an encoder (downsampling path) and a decoder (upsampling path). The encoder extracts features at multiple levels of abstraction, while the decoder uses these features to construct segmented output.

