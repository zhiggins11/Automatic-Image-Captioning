################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
import torchvision.models as models
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embed_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type'] #No implementation of different models yet
    temperature = config_data['generation']['temperature']
    max_length = config_data['generation']['max_length']
    
    
    model = ImageCaptioner(vocab, embed_size, hidden_size, temperature, max_length)
    return model
    
class ImageCaptioner(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, temperature, max_length):
        super().__init__()
        self.encoder = Encoder(embed_size, hidden_size)
        self.decoder = Decoder(vocab, embed_size, hidden_size, temperature, max_length)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        output = self.decoder(features, captions)
        return output
    
    def predict(self, image):
        features = self.encoder(image)
        #print(features)
        prediction = self.decoder.predict(features)
        return prediction
    
    def predict_batch(self, images): #TODO predict as a batch without a loop
        predictions = []
        for i in range(images.size(0)):
            predictions.append(self.predict(image[i]))
        return predictions
        

class Encoder(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super().__init__()
        resnet = models.resnet50(pretrained = True)
        for param in resnet.parameters():
                param.requires_grad = False
        layers = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*layers)
        
        self.fc = nn.Linear(resnet.fc.in_features, hidden_size)       
        

    def forward(self, image):
        x = self.resnet(image)
        y = x.view(x.size(0), -1)
        output = self.fc(y)
        return output

class Decoder(nn.Module):

    def __init__(self, vocab, embed_size, hidden_size, temperature, max_length):
        super().__init__()
        self.vocab = vocab
        self.temperature = temperature
        self.max_length = max_length
        self.embed = nn.Embedding(len(vocab), embed_size)
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, len(vocab))

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        features = torch.unsqueeze(features, 0)
        outputs, (h, c) = self.lstm(embeddings, (features, features)) 
        outputs = self.fc(outputs)
        return outputs, h, c
    
    def predict(self, features): #need to do as a batch #Can probably figure out a way to just call forward a bunch of times here.
        softmax = nn.Softmax(0)
        features = torch.unsqueeze(features, 0)
        h, c = features, features
        current_word = self.vocab.word2idx['<start>']
        caption = [] #['<start>'] -  use if you want to include start
        for i in range(self.max_length):
            current_word = torch.LongTensor([current_word]).to(device)
            embedding = self.embed(current_word)
            embedding = torch.unsqueeze(embedding, 0)
            outputs, (h, c) = self.lstm(embedding, (h,c))
            outputs = self.fc(outputs)
            outputs = outputs.squeeze()
            outputs = outputs/self.temperature
            outputs = softmax(outputs)
            current_word = torch.multinomial(outputs, 1) #current_word = torch.argmax(outputs)
            if current_word == self.vocab.word2idx['<end>']:
                break
            caption.append(self.vocab.idx2word[current_word.item()]) #is current word a tensor or an int?  it is a tensor
        return caption
    
    """def predict_batch(self, features):
        softmax = nn.Softmax(1)
        batch_size = features.size(0)
        start = torch.LongTensor(self.vocab.word2idx['<start>'])
        current_word = start.repeat(batch_size).to(device)
        captions = torch.zeros(batch_size, max_length)
        features = torch.unsqueeze(features, 0)
        h, c = features, features
        #current_word = torch.LongTensor(self.vocab.word2idx['<start>']).to(device)
        for i in range(self.max_length):
            #current_word = torch.LongTensor([current_word]).to(device)
            embedding = self.embed(current_word)
            embedding = torch.unsqueeze(embedding, 0)
            outputs, (h, c) = self.lstm(embedding, (h,c))
            outputs = self.fc(outputs)
            outputs = outputs.squeeze() #is this needed here?
            outputs = outputs/self.temperature
            outputs = softmax(outputs) #outputs should have shape (batch_size, vocab_size)
            current_word = torch.multinomial(outputs, 1)
            caption.append(self.vocab.idx2word[current_word.item()])
            #if current_word == self.vocab.word2idx['<end>']:
            #    break
        return caption"""
            
            
