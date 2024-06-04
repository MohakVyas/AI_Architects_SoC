import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained LSTM model
model = load_model('week 1/lstm_model.h5')

# Define a function for model prediction
def predict(query, num_words):
    # Preprocess the input query
    test = query
    # Generate the predicted sequence
    for _ in range(int(num_words)):
        token_text = tokenizer.texts_to_sequences([test])[0]
        padded_token_test = pad_sequences([token_text], maxlen=230, padding='pre')
        pos = np.argmax(model.predict(padded_token_test))
        for word, index in tokenizer.word_index.items():
          if index == pos:
            test = test + " " + word
    return test



        # Reshape the input sequence
    #     input_seq_reshaped = input_seq.reshape(1, -1, 1)

    #     # Predict the next word
    #     predicted_word = model.predict(input_seq_reshaped)

    #     # Append the predicted word to the input sequence
    #     input_seq = np.append(input_seq, predicted_word)

    # # Postprocess the output sequence
    # predicted_sentence = postprocess_sequence(input_seq)


# Define a function to preprocess the input query
def preprocess_query(query):
    # Tokenize and preprocess the query as required by your model
    # Example: convert query to a numerical sequence
    sequence = np.array([ord(char) for char in query])
    return sequence

# Define a function to postprocess the output sequence
def postprocess_sequence(sequence):
    # Convert the numerical sequence back to text
    sentence = ''.join([chr(int(num)) for num in sequence])
    return sentence

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter input query here..."),
        gr.Slider(minimum=1, maximum=100, step=1, label="Number of words to predict")
    ],
    outputs="text",
    title="LSTM Text Generation",
    description="Enter a query and the number of words to predict. The model will return the complete sentence."
)

# Launch the Gradio app
iface.launch(share=True)
