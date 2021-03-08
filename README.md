# Image Captioning

This is work that I did as part of a group project (with Lingxi Li, Yejun Li, Jiawen Zeng, and Yunyi Zhang) for CSE 251B (Neural Networks).  I benefited from discussions with my partners, but all code here was either given by the instructor or written by me.  Specifically, the instructor gave us a skeleton for project and code for loading the datasets, while I created the model (model_factory.py) and wrote the code for training the model and evaluating it on the validation and tests sets (part of experiment.py).

## Background
This project contains code for building a model which generates captions for images.  It uses ResNet-50 to extract features from images, and then an LSTM to write captions for them.  It uses the COCO dataset.


## Using a pretrained model

Running the `sample.ipynb` notebook will load a trained model and generate captions on test images.  The first cell of this notebook will download the trained model (150 MB) from my Google Drive.  After running the first three cells, each time you runs the last cell, it will choose a random image from the test set and generate a caption for it.  Several of the test images are not available from the COCO dataset website (which is where the code pulls images from so that you do not have to download the entire test set), so if you run into an error here, simply run the last cell again to get a new image.

## Training your own model

If you want to train your own model, you'll need to download some subset of the COCO dataset to use as your training, validation, and test sets.  Then set the first 6 variables under "dataset" in `default.json` to give paths to your images and annotations.  Then: 
1. Set the parameters for your model and experiment in `default.json`.
3. After defining the configuration in `default.json`, simply run `python3 main.py default` to start the experiment.
4. The logs, stats, plots and saved models will be stored in `./experiment_data/default` directory.
5. To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training.

## Files
- main.py: Main driver class
- default.json: File for specifying the configuration of model and experiment
- experiment.py: Main experiment class. Initialized based on config in default.json - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- file_utils.py: Utility functions for handling files 
- caption_utils.py: Utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- sample.ipynb: File for generating sample captions on the test set.

## TODO
1. Add transformations to training images to improve the model's performance on the validation and test sets.
2. Allow model to make caption predicts on a batch to speed up testing.
