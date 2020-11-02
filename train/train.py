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
import tensorflow as tf

from utils import preprocess_text_nonbreaking, subword_tokenize
from utils_train import loss_function, CustomSchedule

from model import Transformer

INPUT_COLUMN = 'input'
TARGET_COLUMN = 'target'
#NUM_SAMPLES = 80000 #40000
#MAX_VOCAB_SIZE = 2**14

#BATCH_SIZE = 64  # Batch size for training.
#EPOCHS = 10  # Number of epochs to train for.
#MAX_LENGTH = 15

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

def get_train_data(training_dir, nonbreaking_in, nonbreaking_out, train_file, nsamples):
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
    # Load the dataset: sentence in english, sentence in spanish 
    df=pd.read_csv(train_filenamepath, sep="\t", header=None, names=[INPUT_COLUMN,TARGET_COLUMN], usecols=[0,1], 
               nrows=nsamples)
    # Preprocess the input data
    input_data=df[INPUT_COLUMN].apply(lambda x : preprocess_text_nonbreaking(x, non_breaking_prefix_en)).tolist()
    # Preprocess and include the end of sentence token to the target text
    target_data=df[TARGET_COLUMN].apply(lambda x : preprocess_text_nonbreaking(x, non_breaking_prefix_es)).tolist()

    return input_data, target_data

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

def main_train(dataset, transformer, n_epochs, print_every=50):
  ''' Train the transformer model for n_epochs using the data generator dataset'''
  losses = []
  accuracies = []
  # In every epoch
  for epoch in range(n_epochs):
    print("Starting epoch {}".format(epoch+1))
    start = time.time()
    # Reset the losss and accuracy calculations
    train_loss.reset_states()
    train_accuracy.reset_states()
    # Get a batch of inputs and targets
    for (batch, (enc_inputs, targets)) in enumerate(dataset):
        # Set the decoder inputs
        dec_inputs = targets[:, :-1]
        # Set the target outputs, right shifted
        dec_outputs_real = targets[:, 1:]
        with tf.GradientTape() as tape:
            # Call the transformer and get the predicted output
            predictions = transformer(enc_inputs, dec_inputs, True)
            # Calculate the loss
            loss = loss_function(dec_outputs_real, predictions)
        # Update the weights and optimizer
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        # Save and store the metrics
        train_loss(loss)
        train_accuracy(dec_outputs_real, predictions)
        
        if batch % print_every == 0:
            losses.append(train_loss.result())
            accuracies.append(train_accuracy.result())
            print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                epoch+1, batch, train_loss.result(), train_accuracy.result()))
            
    # Checkpoint the model on every epoch        
    ckpt_save_path = ckpt_manager.save()
    print("Saving checkpoint for epoch {} in {}".format(epoch+1,
                                                        ckpt_save_path))
    #print("Time for 1 epoch: {} secs\n".format(time.time() - start))
  return losses, accuracies


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
    parser.add_argument('--nsamples', type=int, default=20000, metavar='N',
                        help='number of samples to train (default: 20000)')

    # Data parameters                    
    parser.add_argument('--train_file', type=str, default=None, metavar='N',
                        help='Training data file name')
    parser.add_argument('--non_breaking_in', type=str, default=None, metavar='N',
                        help='Non breaking prefixes for input vocabulary')
    parser.add_argument('--non_breaking_out', type=str, default=None, metavar='N',
                        help='Non breaking prefixes for output vocabulary')

    parser.add_argument('--val-frac', type=float, default=0.1, metavar='N',
                        help='Fraction og data for validation (default: 0.1)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--d_model', type=int, default=64, metavar='N',
                        help='Model dimension (default: 64)')
    parser.add_argument('--ffn_dim', type=int, default=128, metavar='N',
                        help='size of the FFN layer (default: 128)')
    parser.add_argument('--vocab_size', type=int, default=10000, metavar='N',
                        help='size of the vocabulary (default: 10000)')
    parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                        help='number of layers (default: 4)')
    parser.add_argument('--n_heads', type=int, default=6, metavar='N',
                        help='number of heads (default: 6)')
    parser.add_argument('--dropout_rate', type=float, default=0.1, metavar='N',
                        help='Dropout rate (default: 0.1)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    # Load the training data.
    input_data, target_data = get_train_data(args.data_dir, args.non_breaking_in, args.non_breaking_out, args.train_file, args.nsamples)

    # Tokenize and pad the input sequences
    encoder_inputs, tokenizer_inputs, num_words_inputs, sos_token_input, eos_token_input, del_idx_inputs= subword_tokenize(input_data, 
                                                                                                        args.vocab_size, args.max_len)
    # Tokenize and pad the outputs sequences
    decoder_outputs, tokenizer_outputs, num_words_output, sos_token_output, eos_token_output, del_idx_outputs = subword_tokenize(target_data, 
                                                                                                        args.vocab_size, args.max_len)
    # Define a dataset 
    dataset = tf.data.Dataset.from_tensor_slices(
                    (encoder_inputs, decoder_outputs))
    dataset = dataset.shuffle(len(input_data), reshuffle_each_iteration=True).batch(
                    BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Clean the session
    tf.keras.backend.clear_session()
    # Create the Transformer model
    transformer = Transformer(vocab_size_enc=num_words_inputs,
                          vocab_size_dec=num_words_output,
                          d_model=args.d_model,
                          n_layers=args.n_layers,
                          FFN_units=args.ffn_dim,
                          n_heads=args.n_heads,
                          dropout_rate=args.dropout_rate)

    # Define a categorical cross entropy loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction="none")
    # Define a metric to store the mean loss of every epoch
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    # Define a matric to save the accuracy in every epoch
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    # Create the scheduler for learning rate decay
    leaning_rate = CustomSchedule(D_MODEL)
    # Create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

    #Create the Checkpoint 
    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

    #if ckpt_manager.latest_checkpoint:
    #    ckpt.restore(ckpt_manager.latest_checkpoint)
    #    print("Last checkpoint restored.")
    
    # Train the model
    losses, accuracies = main_train(dataset, transformer, args.epochs, 100)

    # Save the while model
    # Save the entire model to a HDF5 file
    transformer.save(os.path.join(args.model_dir, 'transformer.h5'))

	# Save the tokenizers with the vocabularies
    vocabulary_in = os.path.join(args.model_dir, 'in_vocab.pkl')
    with open(vocabulary_in, 'wb') as f:
        pickle.dump(tokenizer_inputs, f)

    vocabulary_out = os.path.join(args.model_dir, 'out_vocab.pkl')
    with open(vocabulary_out, 'wb') as f:
        pickle.dump(tokenizer_outputs, f)
