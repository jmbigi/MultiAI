import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LambdaCallback

# Load and process the dataset
data = open('prediction/python_code_001.txt').read().lower()
chars = sorted(list(set(data)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(data) - maxlen, step):
    sentences.append(data[i: i + maxlen])
    next_chars.append(data[i + maxlen])

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool_)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = True
    y[i, char_indices[next_chars[i]]] = True

# Define the model
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

# Here's the change: adding run_eagerly=True
model.compile(loss='categorical_crossentropy', optimizer=optimizer, run_eagerly=True)

# Helper function to generate text after each epoch
def on_epoch_end(epoch, _):
    print(f'\nGenerating text after epoch: {epoch}')

    start_index = np.random.randint(0, len(data) - maxlen - 1)
    generated = ''
    sentence = data[start_index: start_index + maxlen]
    generated += sentence

    print(f'Generating with seed: "{sentence}"')

    for i in range(400):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char

        generated += next_char

    print(f'Generated: "{generated}"')

generate_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Train the model
model.fit(X, y, batch_size=128, epochs=20, callbacks=[generate_callback])
