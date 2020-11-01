import numpy as np
import re

import pickle
import os

def clean_text(sentences, alpha=False):
    ''' Cleaning process of the text'''
    if alpha:
        # Remove non alphabetic character
        cleaned_text = [''.join([t.lower() for t in text if t.isalpha() or t.isspace()]) for text in sentences]
    else:
        # Simply lower the characters
        cleaned_text = [t.lower() for t in sentences]
    # Remove any emoty string
    cleaned_text = [t for t in cleaned_text if t!='']
    
    return ''.join(cleaned_text)

def one_hot_encode(indices, dict_size):
    ''' Define one hot encode matrix for our sequences'''
    # Creating a multi-dimensional array with the desired output shape
    # Encode every integer with its one hot representation
    features = np.eye(dict_size, dtype=np.float32)[indices.flatten()]
    
    # Finally reshape it to get back to the original array
    features = features.reshape((*indices.shape, dict_size))
            
    return features

def encode_text(input_text, vocab, one_hot = False):
    # Replace every char by its integer value based on the vocabulary
    output = [vocab.get(character,0) for character in input_text]
    
    if one_hot:
    # One hot encode every integer of the sequence
        dict_size = len(vocab)
        return one_hot_encode(output, dict_size)
    else:
        return np.array(output)

