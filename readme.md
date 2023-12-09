# Generative Model using Autoencoders - PyTorch

## Overview

Many of us are aware of Discriminative models, which generally focus on predicting data labels. In contrast, Generative Models help generate output very similar to the input data. 

For example, we have data related to cars that can be used to classify them, but Generative models can learn the data's patterns and generate car features completely different from the input data.

---

## Aim

In this project, we introduce Generative Models, use the PyTorch framework to build Autoencoders on the MNIST dataset, and learn how to use Autoencoders as Generative Models to generate new images of digits.

---

## Tech Stack

- **Language:** Python
- **Libraries:** torch, torchvision, torchinfo, numpy, matplotlib

---

## Approach

1. Introduction to Generative Models
2. Introduction to Autoencoders
3. Building Autoencoders in PyTorch
4. Model training on Google Colab
5. Building Generative Models using Autoencoders

---

## Modular Code Overview

1. `data`: Contains `data_utils.py`, used to download and transform the data.
2. `model`: Contains `autoencoder_decoder_model.py` and the Colab notebook.
3. `train.py`: Contains the code for model training.
4. `run.py`: The main file in which all functions are called.
5. `requirements.txt`: Lists all the required libraries with respective versions. Install them using `pip install -r requirements.txt`. Note: Please use CUDA versions of torch if CUDA is available.

---



