import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the pre-trained LSTM model
model = load_model('D:/lstm_model.h5')

# Load the tokenizer
with open('D:/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define a function for model prediction
def predict(query, num_words):
    test = query
    # Generate the predicted sequence
    for _ in range(int(num_words)):
        token_text = tokenizer.texts_to_sequences([test])[0]
        padded_token_test = pad_sequences([token_text], maxlen=230, padding='pre')
        pos = np.argmax(model.predict(padded_token_test), axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == pos:
                test = test + " " + word
                break
    return test

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
