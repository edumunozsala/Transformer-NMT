import argparse
import json
import sys
import sagemaker_containers

import math
import os
import gc
import time
import re
import pandas as pd
import numpy as np

from utils import preprocess_text_nonbreaking, subword_tokenize

from model import Transformer

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNModel(model_info['vocab_size'], model_info['embedding_dim'], model_info['hidden_dim'], model_info['n_layers'], model_info['drop_rate'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'char_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.char2int = pickle.load(f)

    word_dict_path = os.path.join(model_dir, 'int_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.int2char = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def batch_generator_sequence(features_seq, label_seq, batch_size, seq_len):
    """Generator function that yields batches of data (input and target)

    Args:
        features_seq: sequence of chracters, feature of our model.
        label_seq: sequence of chracters, the target label of our model
        batch_size (int): number of examples (in this case, sentences) per batch.
        seq_len (int): maximum length of the output tensor.

    Yields:
        x_epoch: sequence of features for the epoch
        y_epoch: sequence of labels for the epoch
    """
    # calculate the number of batches we can supply
    num_batches = len(features_seq) // (batch_size * seq_len)
    if num_batches == 0:
        raise ValueError("No batches created. Use smaller batch size or sequence length.")
    # calculate effective length of text to use
    rounded_len = num_batches * batch_size * seq_len
    # Reshape the features matrix in batch size x num_batches * seq_len
    x = np.reshape(features_seq[: rounded_len], [batch_size, num_batches * seq_len])
    
    # Reshape the target matrix in batch size x num_batches * seq_len
    y = np.reshape(label_seq[: rounded_len], [batch_size, num_batches * seq_len])
    
    epoch = 0
    while True:
        # roll so that no need to reset rnn states over epochs
        x_epoch = np.split(np.roll(x, -epoch, axis=0), num_batches, axis=1)
        y_epoch = np.split(np.roll(y, -epoch, axis=0), num_batches, axis=1)
        for batch in range(num_batches):
            yield x_epoch[batch], y_epoch[batch]
        epoch += 1

def _get_train_data(batch_size, maxlen, training_dir, nonbreaking_in, nonbreaking_out):
    print("Get the train data loader.")
    # Load the nonbreaking files
    with open(os.path.join(training_dir, nonbreaking_in), 
          mode = "r", encoding = "utf-8") as f:
    non_breaking_prefix_en = f.read()
    with open(os.path.join(training_dir, nonbreaking_out), 
          mode = "r", encoding = "utf-8") as f:
    non_breaking_prefix_es = f.read()

    non_breaking_prefix_en = non_breaking_prefix_en.split("\n")
    non_breaking_prefix_en = [' ' + pref + '.' for pref in non_breaking_prefix_en]
    non_breaking_prefix_es = non_breaking_prefix_es.split("\n")
    non_breaking_prefix_es = [' ' + pref + '.' for pref in non_breaking_prefix_es]
    # Load the training data
    
    with open(os.path.join(training_dir, "input_data.pkl"), "rb") as fp:   # Unpickling
       train_data = pickle.load(fp)

    #train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    # The input sequence, from 0 to len-1
    input_seq=train_data[:-1]
    # The target sequence, from 1 to len
    target_seq=train_data[1:]

    # Create a batch generator for training along the input data
    batch_data = batch_generator_sequence(input_seq, target_seq, batch_size, maxlen)
    # Calculate the number of batches to complete an epoch (all training data)
    num_batches = len(input_seq) // (batch_size*maxlen)

    return batch_data, num_batches

def set_device():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    
    return device

def train_main(model, optimizer, loss_fn, batch_data, num_batches, val_batches, batch_size, seq_len, n_epochs, clip_norm, device):
    # Training Run
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        # Store the loss in every batch iteration
        epoch_losses =[]
        # Init the hidden state
        hidden = model.init_state(device, batch_size)
        # Train all the batches in every epoch
        for i in range(num_batches-val_batches):
            # Get the next batch data for input and target
            input_batch, target_batch = next(batch_data)
            # Onr hot encode the input data
            input_batch = one_hot_encode(input_batch, model.vocab_size)
            # Tranform to tensor
            input_data = torch.from_numpy(input_batch)
            target_data = torch.from_numpy(target_batch)
            # Create a new variable for the hidden state, necessary to calculate the gradients
            hidden = tuple(([Variable(var.data) for var in hidden]))
            # Move the input data to the device
            input_data = input_data.to(device)
            # Set the model to train and prepare the gradients
            model.train()
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            # Pass Fordward the RNN
            output, hidden = model(input_data, hidden)
            output = output.to(device)
            # Move the target data to the device
            target_data = target_data.to(device)
            target_data = torch.reshape(target_data, (batch_size*seq_len,))
            # Calculate the loss
            loss = loss_fn(output, target_data.view(batch_size*seq_len))
            # Save the loss
            epoch_losses.append(loss.item()) #data[0]
            # Does backpropagation and calculates gradients
            loss.backward()
            # clip gradient norm
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            # Updates the weights accordingly
            optimizer.step()
    
        # Now, when the epoch is finished, evaluate the model on validation data
        model.eval()
        val_hidden = model.init_state(device, batch_size)
        val_losses = []
        for i in range(val_batches):
            # Get the next batch data for input and target
            input_batch, target_batch = next(batch_data)
            # Onr hot encode the input data
            input_batch = one_hot_encode(input_batch, model.vocab_size)
            # Tranform to tensor
            input_data = torch.from_numpy(input_batch)
            target_data = torch.from_numpy(target_batch)
            # Create a new variable for the hidden state, necessary to calculate the gradients
            hidden = tuple(([Variable(var.data) for var in val_hidden]))
            # Move the input data to the device
            input_data = input_data.to(device)
            # Pass Fordward the RNN
            output, hidden = model(input_data, hidden)
            output = output.to(device)
            # Move the target data to the device
            target_data = target_data.to(device)
            target_data = torch.reshape(target_data, (batch_size*seq_len,))
            loss = loss_fn(output, target_data.view(batch_size*seq_len))
            # Save the loss
            val_losses.append(loss.item()) #data[0]

        model.train()                  
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print('Time: {:.4f}'.format(time.time() - start_time), end=' ')
        print("Train Loss: {:.4f}".format(np.mean(epoch_losses)), end=' ')
        print("Val Loss: {:.4f}".format(np.mean(val_losses)))
        
    return epoch_losses

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-len', type=int, default=60, metavar='N',
                        help='input max sequence length for training (default: 60)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='learning rate to train (default: 0.01)')
    parser.add_argument('--clip_norm', type=int, default=5, metavar='N',
                        help='Gradients clip norm (default: 5)')
    parser.add_argument('--val-frac', type=float, default=0.1, metavar='N',
                        help='Fraction og data for validation (default: 0.1)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=38, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=38, metavar='N',
                        help='size of the vocabulary (default: 28)')
    parser.add_argument('--n_layers', type=int, default=1, metavar='N',
                        help='number of layers (default: 1)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader, num_batches = _get_train_data_loader(args.batch_size, args.max_len, args.data_dir)
    # Calculate the validation batches
    val_batches = int(num_batches*args.val_frac)
    # Build the model.
    # Instantiate the model with hyperparameters
    model = RNNModel(args.vocab_size,args.vocab_size, args.hidden_dim, args.n_layers).to(device)

    # Load the dictionaries
    with open(os.path.join(args.data_dir, "char_dict.pkl"), "rb") as f:
        model.char2int = pickle.load(f)

    with open(os.path.join(args.data_dir, "int_dict.pkl"), "rb") as f:
        model.int2char = pickle.load(f)

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size
    ))

    # Train the model.
    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_main(model, optimizer, criterion, train_loader, num_batches, val_batches, args.batch_size, args.max_len, args.epochs, args.clip_norm, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'n_layers': args.n_layers,
            'embedding_dim': args.vocab_size,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
            'drop_rate': 0.2
        }
        torch.save(model_info, f)

	# Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'char_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.char2int, f)

    word_dict_path = os.path.join(args.model_dir, 'int_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.int2char, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)
