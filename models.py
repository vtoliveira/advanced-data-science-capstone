import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import preprocess_tweet, preprocess_text, tokenize_custom, pad_features

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, vocab_to_int,  drop_prob=0.5, train_on_gpu=True):
        """
        Initialize the model by setting up the layers.
        """
        
        super(SentimentRNN, self).__init__()
        self.vocab_to_int = vocab_to_int
        self.embedding_dim = embedding_dim
        self.train_on_gpu = train_on_gpu
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # define all layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        
        embeds = self.embedding(x)
        #print('embeds', embeds.size())
        lstm_out, hidden = self.lstm(embeds, hidden)
        #print('lstm1',lstm_out.size())
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        #print('lstm2',lstm_out.size())
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # reshape to be batch_size first
        #print('out', out.size())
        out = out.view(batch_size, -1)
        #print('out', out.size())
        out = out[:, -3:] # get last batch of labels
        # return last sigmoid output and hidden state
        
        #print('out', out.size())
        #out = 1
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        #if (train_on_gpu):
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), 
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        #else:
        #    hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
        #              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
    
    def train_data(self, train_loader, valid_loader, batch_size, criterion, optimizer, saving_path, lr = 0.01, clip = 5, num_epochs = 5):
        """ train the net """

        self.train_history = []
        self.valid_history = []

        valid_loss_min = np.Inf

        if(self.train_on_gpu):
            self.cuda()

        self.train()

        for e in range(num_epochs):
            train_loss = 0
            valid_loss = 0
            
            h = self.init_hidden(batch_size)
            self.train()
            for inputs, labels in train_loader:
                if(self.train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                inputs = inputs.type(torch.cuda.LongTensor)
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                self.zero_grad()

                # get the output from the model
                output, h = self(inputs, h)

                # calculate the loss and perform backprop
                loss = criterion(output, labels)
                loss.backward()
                
                # recording training loss
                train_loss += loss.item()


                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.parameters(), clip)

                # optimizer the weights
                optimizer.step()

            val_h = self.init_hidden(batch_size)
            self.eval()
            test_acc = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    if(self.train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    inputs = inputs.type(torch.cuda.LongTensor)

                    val_h = tuple([each.data for each in val_h])

                    output, val_h = self(inputs, val_h)

                    val_loss = criterion(output, labels)
                    valid_loss += val_loss.item()

                    preds = output.argmax(dim=1)
                    test_acc += np.sum(preds.eq(labels).cpu().numpy()) / len(inputs)

            self.valid_history.append(valid_loss / len(valid_loader))
            self.train_history.append(train_loss / len(train_loader))

            print("Epoch: {}/{}...".format(e+1, num_epochs),
            "Training Loss: {:.6f}...".format(train_loss / len(train_loader)),
             "Valid Loss: {:.6f}...".format(valid_loss / len(valid_loader)),
             "Valid Accuracy: {:.3f}".format((test_acc / len(valid_loader))))
            
            valid_loss = valid_loss / len(valid_loader)
            if valid_loss <= valid_loss_min:
                valid_loss_min = valid_loss
                print('Saving../')
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_losses':  self.train_history,
                    'valid_losses':self.valid_history
                    }, saving_path)
    
    def test(self, test_loader, batch_size, criterion, print_baseline = True):
        h = self.init_hidden(batch_size)
        self.eval()
        self.predictions = []
        self.labels = []

        test_acc = 0
        test_loss = 0
        baseline = np.zeros((self.output_size))

        for inputs, labels in test_loader:
            if(self.train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs = inputs.type(torch.cuda.LongTensor)
            h = tuple([each.data for each in h])

            output, h = self(inputs, h)
            loss = criterion(output, labels)

            test_loss += loss

            preds = output.argmax(dim =1)
            self.predictions.append(preds)
            self.labels.append(labels)

            test_acc += np.sum(preds.eq(labels).cpu().numpy()) / len(inputs)


        print("Test loss: {:.3f}".format(test_loss / len(test_loader)),
                "Test accuracy: {:.3f}".format(test_acc / len(test_loader))) 

    def predict(self, tweet, seq_length):
        tweet = preprocess_tweet(tweet, punctuation = True)

        tweet = preprocess_text(tweet)

        tokens = [tokenize_custom(tweet, self.vocab_to_int)]

        features = pad_features(tokens, seq_length=seq_length)

        self.cuda()
        with torch.no_grad():
            h = self.init_hidden(1)
            output, h =  self(torch.from_numpy(features).type(torch.cuda.LongTensor), h)

            softmax = nn.Softmax(dim=1)
        return softmax(output).cpu().numpy()