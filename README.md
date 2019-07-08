# dog-breed-classifier-pytorch
This repository contains a project I completed using PyTorch in the language Python. First, I built a full neural network containing convolutional and fully connected linear layers to classify the breed of a dog given an image as input. This was not as successful as I would have liked with an accuracy of only 11%, so I implemented transfer learning. Using the pretrained VGG16 image classification model, I reworked and trained the fully connected linear layers to give the network the number of outputs desired. After training for 20 epochs, I had an accuracy of 62%.

---

## Installation
The code requires the following packages: numpy, glob from glob, matplotlib, cv2, tqdm from tqdm, torch, torchvision, Image from PIL, os, and ImageFile from PIL.

The Jupyter notebook or Python file can be donloaded and run in an environment with the installed packages. The easiest way to set up and run the code in an environment is using [Anaconda](https://www.anaconda.com/distribution/). Anaconda is a free, open-source platorm to manage libraries, dependencies, and environments.

---

#### Note
This project is part of the Udacity Deep Learning Nanodegree.
