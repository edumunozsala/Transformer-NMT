import torch
from torch import nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, n_layers, drop_rate=0.2):
        
        super(RNNModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate
        self.char2int = None
        self.int2char = None


        #Defining the layers
        # Define the encoder as an Embedding layer
        #self.encoder = nn.Embedding(vocab_size, embedding_size)
            
        # Dropout layer
        self.dropout = nn.Dropout(drop_rate)
        # RNN Layer
        self.rnn = nn.LSTM(embedding_size, hidden_dim, n_layers, dropout=drop_rate, batch_first = True)
        # Fully connected layer
        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, state):
        
        # input shape: [batch_size, seq_len, embedding_size]
        # Apply the embedding layer and dropout
        #embed_seq = self.dropout(self.encoder(x))
            
        #print('Input RNN shape: ', embed_seq.shape)
        # shape: [batch_size, seq_len, embedding_size]
        rnn_out, state = self.rnn(x, state)
        #print('Out RNN shape: ', rnn_out.shape)
        # rnn_out shape: [batch_size, seq_len, rnn_size]
        # hidden shape: [2, num_layers, batch_size, rnn_size]
        rnn_out = self.dropout(rnn_out)

        # shape: [seq_len, batch_size, rnn_size]
        #logits = self.decoder(rnn_out.view(-1, rnn_out.size(2)))
        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim)

        logits = self.decoder(rnn_out)
        # output shape: [seq_len * batch_size, vocab_size]
        #print('Output model shape: ', logits.shape)
        return logits, state
    
    def init_state(self, device, batch_size=1):
        """
        initialises rnn states.
        """
        #return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)))
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden
    
    def predict(self, input):
        # input shape: [seq_len, batch_size]
        logits, hidden = self.forward(input)
        # logits shape: [seq_len * batch_size, vocab_size]
        # hidden shape: [2, num_layers, batch_size, rnn_size]
        probs = F.softmax(logits)
        # shape: [seq_len * batch_size, vocab_size]
        probs = probs.view(input.size(0), input.size(1), probs.size(1))
        # output shape: [seq_len, batch_size, vocab_size]
        return probs, hidden
