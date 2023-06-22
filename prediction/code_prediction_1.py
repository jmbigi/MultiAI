import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
import glob

directory = 'prediction/pythons/*.py'

# Read and process the files
data = ""
for file_path in glob.glob(directory):
    with open(file_path) as file:
        data += file.read().lower()

# Create a sorted list of the unique characters in the data
chars = sorted(list(set(data)))

# Create dictionaries to map characters to indices and vice versa
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Prepare the dataset
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(data) - maxlen, step):
    sentences.append(data[i: i + maxlen])
    next_chars.append(data[i + maxlen])

# One-hot encode the data
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool_)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = True
    y[i, char_indices[next_chars[i]]] = True

# Define the model
model = Sequential()

# Add LSTM layer with 128 units
model.add(LSTM(128, input_shape=(maxlen, len(chars))))

# Add dropout layer to prevent overfitting
model.add(Dropout(0.2))

# Add output layer
model.add(Dense(len(chars), activation='softmax'))

# Compile the model
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, run_eagerly=True)

# Define callback function to generate text after each epoch
def on_epoch_end(epoch, _):
    print()
    print(f'\nGenerating text after epoch: {epoch}')

    # Select a random starting point for generation
    start_index = np.random.randint(0, len(data) - maxlen - 1)
    generated = ''
    sentence = data[start_index: start_index + maxlen]
    generated += sentence

    print(f'Generating with seed: "{sentence}"')

    # Generate characters
    for i in range(80):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char

        generated += next_char

    print(f'Generated: "{generated}"')
    print()

# Instantiate the LambdaCallback for generation
generate_callback = LambdaCallback(on_epoch_end=on_epoch_end, save_freq=5)

# Set up checkpoint path for saving weights
checkpoint_path = "training/cp.ckpt"

# Create ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

# Train the model
model.fit(X, y, batch_size=128, epochs=20, callbacks=[generate_callback, checkpoint_callback])
