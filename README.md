# Attention is all you need: Discovering the Transformer model

# Neural machine traslation using a Transformer model

In this repository we will develop and demystify the relevant artifacts in the paper "Attention is all you need" (Vaswani, Ashish & Shazeer, Noam & Parmar, Niki & Uszkoreit, Jakob & Jones, Llion & Gomez, Aidan & Kaiser, Lukasz & Polosukhin, Illia. (2017)). This paper was a more advanced step in the use of the attention mechanism being the main basis for a model called **Transformer**. The most famous current models that are emerging in NLP tasks consist of dozens of transformers or some of their variants, for example, GPT-3 or BERT.

We will describe the components of this model, analyze their operation and build a simple model that we will apply to a small-scale NMT problem (Neural Machine Translation). To read more about the problem that we will address and to know how the basic attention mechanism works, I recommend you to read my previous post [ "A Guide on the Encoder-Decoder Model and the Attention Mechanism"](https://medium.com/better-programming/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb) and check the [ repository](https://github.com/edumunozsala/NMT-encoder-decoder-Attention).

**On development**
*Once we have our model developed and tested, we create a second notebook to train the model in Amazon SageMaker, using a more powerful compute instance for longer training. Then we deploy our trained model as an endpoint on an instance in SageMaker. We can invoke this endpoint to make predictions on our model, and built a complete sentence from an initial string.*

## The data set

For this exercise we will use pairs of simple sentences, the source in English and target in Spanish, from the Tatoeba project where people contribute adding translations every day. This is the [link](http://www.manythings.org/anki/) to some traslations in different languages. There you can download the Spanish - English spa_eng.zip file, it contains 124457 pairs of sentences. The data is also available in the data folder in this repo.

We use a list of **non breaking prefixes** to avoid the tokenizer to split or break words including that prefixes. Inm our example we do not want to remove some the dot for some well-konw words.You can find non breaking prefixes for many languages in the Kaggle website: 

https://www.kaggle.com/nltkdata/nonbreaking-prefixes/activity


The text sentences are almost clean, they are simple plain text, so we only need to remove dots that are not a end of sentence symbol and duplicated white spaces. 

## Problem description

*Machine translation (MT) is the task of automatically converting source text in one language to text in another language. Given a sequence of text in a source language, there is no one single best translation of that text to another language. This is because of the natural ambiguity and flexibility of human language. This makes the challenge of automatic machine translation difficult, perhaps one of the most difficult in artificial intelligence.*

From the above we can deduce that NMT is a problem where we process an input sequence to produce an output sequence, that is, a sequence-to-sequence (seq2seq) problem. Specifically of the many-to-many type, sequence of several elements both at the input and at the output, and the encoder-decoder architecture for recurrent neural networks is the standard method.

## Content
This repository contains the next source code file:
- Transformer-NMT-en-es: This notebook shows how to download and preprocess the text data, create a batch data generator for sequences of data, define and build all the components in a **Transformer using the self attention mechanism**. We describe the attention mechanism, the encoder and decoder blocks and build the encoder, the decoder and the Transformer and train it for our problem.

**On development**
- Transformer-NMT-en-es-SageMaker: It is a demo on how to train a ML model in the framework Amazon SageMaker (using the model in the previous notebook). This model is very simple so you do not really need to launch a training job on SageMaker but it is intended for educational purposes. We also deploy our model as a service and make some predictions. This notebook requieres the folders:
    - train: here is where the training script is located. It also includes the model.py with the definition of our Transformer Class and a utils.py with some helper functions to preprocess the text.
    - serve: this folder is needed to deploy the model, it contains a model.py and utils.py, the same as before, and a predict.py with the lines of code to make predictions on the model. 

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a public GNU License.