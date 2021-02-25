# Image Captioning

This is work that I did as part of a group project (with Lingxi Li, Yejun Li, Jiawen Zeng, and Yunyi Zhang) for CSE 251B (Neural Networks).  I benefited from discussions with my partners, but all code here was either given by the instructor or written by me.  Specifically, I created the model (model_factory3.py) and wrote the code for training the model and evaluating it on the validation and tests sets (part of experiment.py), but everything else was given by the instructor (including all data preprocessing).

## Usage

1. First, run get_datasets.ipynb to get the training, validation, and test data.
2. Set the parameters for your model and experiment in 'default.json'. See `default.json` to see the structure and available options.
3. After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
4. The logs, stats, plots and saved models will be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
5. To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory3.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace

## TODO
    Upload a trained model and include a function for showing a few sample captions on the test set
