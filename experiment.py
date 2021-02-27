################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import nltk
from torchvision import transforms
from datetime import datetime

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)
        
        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []

        self.__best_loss = 100000
        self.__best_model = None

        self.__model = get_model(config_data, self.__vocab).to(device)

        self.__criterion = nn.CrossEntropyLoss().to(device)
        params = list(self.__model.decoder.parameters()) + list(self.__model.encoder.fc.parameters())
        self.__optimizer = optim.Adam(params = params, lr = config_data['experiment']['learning_rate'])

        # Load Experiment Data if available
        #self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            print("loading the model")
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt')) 
            self.__model.load_state_dict(state_dict)(['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    def to_sentence_tensor(self, caption):
        sentence = []
        for i in range(len(caption)):
            sentence.append(self.__vocab.idx2word[caption[i].item()])
        return sentence
    
    def make_pic(self, ):
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        z = tensor * std.view(3, 1, 1)
        z = z + mean.view(3, 1, 1)
        z = torch.reshape(z, (3,256,256))
        z = z.to("cpu")
        image = transforms.ToPILImage(mode='RGB')(z)
        display(image)
    
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train(epoch)
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    def __train(self):
        self.__model.train()
        training_loss = 0
        counter = 0
        
        for i, (images, captions, _) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            if torch.cuda.is_available():
                images = images.to(torch.device("cuda"))
                captions = captions.to(torch.device("cuda"))
            outputs, h, c  = self.__model(images, captions[:, :-1])
            loss = self.__criterion(torch.transpose(outputs, 1, 2), captions[:,1:]) 
            loss.backward()
            if i%10 == 0:
                print("Epoch {}, Batch {} Training Loss: {}".format(self.__current_epoch+1, i, loss.item()))
            training_loss += loss.item() 
            counter += 1
            self.__optimizer.step()

        training_loss /= counter
        
        return training_loss

    def __val(self):
        self.__model.eval()
        val_loss = 0
        counter = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images, captions = images.to(device), captions.to(device)
                outputs, h, c = self.__model(images, captions[:,:-1])
                val_loss += self.__criterion(torch.transpose(outputs,1,2), captions[:,1:]).item()
                counter += 1
        val_loss /= counter
        if val_loss < self.__best_loss:
            self.__best_loss = val_loss
            self.__best_model = self.__model
        return val_loss

    def test(self):
        self.__model.eval()
        test_loss = 0
        bleu1_score = 0
        bleu4_score = 0
        counter = 0

        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images, captions = images.to(device), captions.to(device)
                batch_size = images.size(0)
                
                outputs, h, c = self.__best_model(images, captions[:,:-1])
                test_loss += self.__criterion(torch.transpose(outputs,1,2), captions[:,1:]).item()
                
                for i in range(batch_size): #Really, we should do this as a batch
                    reference_captions = [nltk.tokenize.word_tokenize(ann['caption'].lower()) for ann in self.__coco_test.imgToAnns[img_ids[i]]]
                    predicted_caption = self.__best_model.predict(images[i].unsqueeze(0)) #Why do we need to unsqueeze here?
                    bleu1_score += bleu1(reference_captions, predicted_caption)/batch_size
                    bleu4_score += bleu4(reference_captions, predicted_caption)/batch_size
         
                counter += 1
                
            test_loss /= counter
            bleu1_score /= counter
            bleu4_score /= counter

        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss, bleu1_score, bleu4_score)
        self.__log(result_str)

        return test_loss, bleu1, bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)
        torch.save(self.__best_model.state_dict(), 'best_model.pt')

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
