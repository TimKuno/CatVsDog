# CatVsDog
Cat vs Dog classification using a convolutional neural network (CNN) and PyTorch.
The Kaggle [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) 
competition consists of a very large set of cats and dogs.
This means it is a binary classification problem with 2 classes, wich is a good use case for a CNN.

<br>

The goal of this competition was to predict how likely the test images show a dog.
This means that the test images are unlabeled, which makes it impossible to check the predictions of the model.
As a workaround for this problem, I moved 20% of the images from the train folder to the test folder.
Since the train folder contains 25.000 images, this means a train/ test split fo 20.000 to 5.000.
It still leaves enough images for training.

# Setup
Code developed and tested on the following environment:
- windows 10
- python 3.9.19
- torch 1.13.1
- torchvision 0.14.1
- Pillow 9.4.0

Donwload the train.zip from the [Kaggle Competition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data?select=train.zip) 
site and unzip it into the project folder.
Additionally create the folder test in the project folder.
To run the program it is enough to execute the main.py file. 
Either in any IDE like PyCharm or the terminal.