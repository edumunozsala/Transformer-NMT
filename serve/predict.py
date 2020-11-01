import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import numpy as np
import torch
import torch.nn as nn

from model import RNNModel

from utils import clean_text, encode_text, one_hot_encode

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

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        # Extract the desired length of the output string
        sep_pos= data.find('-')
        if sep_pos > 0:
            length = data[:sep_pos]
            initial_string = data[sep_pos+1:]
        return (length,initial_string)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def sample_from_probs(probs, top_n=10):
    """
    truncated weighted random choice.
    """
    _, indices = torch.sort(probs)
    # set probabilities after top_n to 0
    probs[indices.data[:-top_n]] = 0
    #print(probs.shape)
    sampled_index = torch.multinomial(probs, 1)
    return sampled_index

def predict_probs(model, hidden, character, vocab, device):
    # One-hot encoding our input to fit into the model
    character = np.array([[vocab[c] for c in character]])
    character = one_hot_encode(character, len(vocab))
    character = torch.from_numpy(character)
    character = character.to(device)
    
    with torch.no_grad():
        out, hidden = model(character, hidden)

    prob = nn.functional.softmax(out[-1], dim=0).data

    return prob, hidden

def predict_old(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.char2int is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    model.eval() # eval mode
    start = input_data.lower()
    # Clean the text as the text used in training 
    start = clean_text(start, True)
    # Encode the text
    #encode_text(start, model.vocab, one_hot = False):
    # First off, run through the starting characters
    chars = [ch for ch in start]

    state = model.init_state(device, 1)
    probs, state = predict_probs(model, state, chars, model.vocab, device)
    #print(probs.shape)
    #probs = torch.transpose(probs, 0, 1)
    next_index = sample_from_probs(probs[-1].squeeze(), top_n=3)
    
    return next_index.data[0].item()

def predict_fn(input_data, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.char2int is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # TODO: Process input_data so that it is ready to be sent to our model.
    #       You should produce two variables:
    #         data_X   - A sequence of length 500 which represents the converted review
    #         data_len - The length of the review
    # Extract the input data and the desired length
    out_len, start = input_data
    out_len = int(out_len)

    model.eval() # eval mode
    start = start.lower()
    # Clean the text as the text used in training 
    start = clean_text(start, True)
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Init the hidden state
    state = model.init_state(device, 1)

    # Warm up the initial state, predicting on the initial string
    for ch in chars:
        #char, state = predict(model, ch, state, top_n=top_k)
        probs, state = predict_probs(model, state, ch, model.char2int, device)
        next_index = sample_from_probs(probs, 5)

    # Include the last char predicted to the predicted output
    chars.append(model.int2char[next_index.data[0]])   
    # Now pass in the previous characters and get a new one
    for ii in range(size-1):
        #char, h = predict_char(model, chars, vocab)
        probs, state = predict_probs(model, state, chars[-1], model.char2int, device)
        next_index = sample_from_probs(probs, 5)
        # append to sequence
        chars.append(model.int2char[next_index.data[0]])

    # Join all the chars    
    #chars = chars.decode('utf-8')
    return ''.join(chars)
    