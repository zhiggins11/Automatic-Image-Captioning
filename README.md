# Image Captioning

This project is work that I did for a course on deep learning (CSE 251B) at UCSD.  It contains a neural network which generates captions for images.  This neural network uses ResNet-50 to extract features from images, and then an LSTM to write captions for them.  It was trained on the [COCO dataset](https://cocodataset.org/#home).

## Sample results

Here are some captions which were generated by the trained model.

![Sample 0](/testimages/test0.PNG)

![Sample 2](/testimages/test2.PNG)

![Sample 3](/testimages/test3.PNG)

![Sample 4](/testimages/test4.PNG)

![Sample 5](/testimages/test5.PNG)

![Sample 8](/testimages/test8.PNG)

![Sample 9](/testimages/test9.PNG)

## Numerical results

I experimented with different network architectures and hyperparameters to build the best model possible (specific details available upon request).  The best model obtained a BLEU score of 0.656 on the test set.

## Using a pretrained model

Running the `sample.ipynb` notebook will load a trained model and generate captions on test images.  The first cell of this notebook will download the trained model (150 MB) from my Google Drive.  After running the first three cells, each time you run the last cell, it will choose a random image from the test set and generate a caption for it.  Several of the test images are not available as individual images from the COCO dataset website (which is where the code pulls images from so that you do not have to download the entire test set), so if you run into an error here, simply run the last cell again to get a new image.

## Training your own model

If you want to train your own model, you'll need to download some subset of the COCO dataset to use as your training, validation, and test sets.  Then set the first 6 variables under "dataset" in `default.json` to give paths to your images and annotations.  Then: 
1. Set the parameters for your model and experiment in `default.json`.
3. After defining the configuration in `default.json`, simply run `python3 main.py default` to start the experiment.
4. The logs, stats, plots and saved models will be stored in `./experiment_data/default` directory.
5. To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training.

## Files
Several of the files for importing data from the COCO dataset were given by the course instructors:
- dataset_factory: dataset builder based on configuration specified in default.json
- file_utils.py: Utility functions for handling files 
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` for the Coco Dataset

I implemented the rest of the files:
- default.json: File for specifying the configuration of model and experiment
- model_factory.py: Model builder based on configuration specified in default.json
- experiment.py: Main experiment class. Initialized based on config in default.json - takes care of training, saving stats and plots, logging and resuming experiments.
- main.py: Main driver class
- sample.ipynb: File for generating sample captions on the test set.
- caption_utils.py: Utility functions to generate bleu scores

<!---## TODO
1. Add transformations to training images to improve the model's performance on the validation and test sets.
2. Allow model to make caption predicts on a batch to speed up testing.
-->
