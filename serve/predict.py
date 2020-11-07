import tensorflow as tf


def predict(model,inp_sentence, tokenizer_in, tokenizer_out, target_max_len,sos_token_input,eos_token_input,sos_token_output,
           eos_token_output):
    # Tokenize the input sequence using the tokenizer_in
    inp_sentence = sos_token_input + tokenizer_in.encode(inp_sentence) + eos_token_input
    enc_input = tf.expand_dims(inp_sentence, axis=0)

    # Set the initial output sentence to sos
    out_sentence = sos_token_output
    # Reshape the output
    output = tf.expand_dims(out_sentence, axis=0)

    # For max target len tokens
    for _ in range(target_max_len):
        # Call the transformer and get the logits 
        predictions = model(enc_input, output, False) #(1, seq_length, VOCAB_SIZE_ES)
        # Extract the logists of the next word
        prediction = predictions[:, -1:, :]
        # The highest probability is taken
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        # Check if it is the eos token
        if predicted_id == eos_token_output:
            return tf.squeeze(output, axis=0)
        # Concat the predicted word to the output sequence
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def translate(model,sentence, tokenizer_inputs, tokenizer_outputs, max_len, sos_token_input, eos_token_input,sos_token_output,
           eos_token_output ):
    # Get the predicted sequence for the input sentence
    output = predict(model,sentence, tokenizer_inputs, tokenizer_outputs, max_len, sos_token_input,eos_token_input,sos_token_output,
           eos_token_output).numpy()
    # Transform the sequence of tokens to a sentence
    predicted_sentence = tokenizer_outputs.decode(
        [i for i in output if i < sos_token_output]
    )

    return predicted_sentence
