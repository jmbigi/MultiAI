import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
import glob
import re

directory = 'prediction/pythons/*.py'

# Read and process the files
data_code = ""
data_comments = ""
for file_path in glob.glob(directory):
    with open(file_path) as file:
        content = file.read()
        data_code += content.lower()
        comments = re.findall(r"#.*$", content, flags=re.MULTILINE)
        data_comments += ' '.join(comments).lower()

# Prepare the input and target sequences
input_texts = data_comments.split('\n')

# Create a sorted list of unique characters in the dataset
all_chars = sorted(list(set(''.join(input_texts))))

# Create dictionaries to map characters to indices and vice versa
char_indices = {char: i for i, char in enumerate(all_chars)}
indices_char = {i: char for i, char in enumerate(all_chars)}

# Determine the maximum sequence length
max_sequence_length = max([len(txt) for txt in input_texts])

# Prepare the input and target data
encoder_input_data = np.zeros((len(input_texts), max_sequence_length, len(all_chars)), dtype=np.float32)
decoder_input_data = np.zeros((len(input_texts), max_sequence_length, len(all_chars)), dtype=np.float32)
decoder_target_data = np.zeros((len(input_texts), max_sequence_length, len(all_chars)), dtype=np.float32)

for i in range(len(input_texts)):
    for t, char in enumerate(input_texts[i]):
        encoder_input_data[i, t, char_indices[char]] = 1.0
        decoder_input_data[i, t, char_indices[char]] = 1.0

        if t > 0:
            decoder_target_data[i, t - 1, char_indices[char]] = 1.0

# Define the model architecture
encoder_inputs = Input(shape=(None, len(all_chars)))
encoder = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, len(all_chars)))
decoder = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(len(all_chars), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=50)

# Save the trained model
model.save('code_translation_model.h5')
