import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("data/data.csv")

X = df["text"].astype(str)
y = df["label"]

# ==========================
# TOKENIZATION
# ==========================
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words,oov_token="<OOV>")
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(
    sequences,
    maxlen=max_len,
    padding='post'
)

# ==========================
# SPLIT DATA
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# BUILD LSTM
# ==========================
model = Sequential([
    Embedding(max_words,128,input_length=max_len),

    LSTM(
        128,
        return_sequences=False
    ),

    Dropout(0.3),

    Dense(64,activation='relu'),

    Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test,y_test)

print(f"Accuracy : {acc:.4f}")

# ==========================
# SAVE MODEL
# ==========================
model.save("model/lstm_model.h5")

with open("model/tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

with open("model/max_len.pkl","wb") as f:
    pickle.dump(max_len,f)

print("Model Saved")