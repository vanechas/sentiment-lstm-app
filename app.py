import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =================================
# PAGE CONFIG
# =================================
st.set_page_config(
    page_title="LSTM Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# =================================
# LOAD MODEL
# =================================
@st.cache_resource
def load_assets():

    model = load_model("model/lstm_model.h5")

    with open("model/tokenizer.pkl","rb") as f:
        tokenizer = pickle.load(f)

    with open("model/max_len.pkl","rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_assets()

# =================================
# HEADER
# =================================
st.title("🤖 Sentiment Analysis using LSTM")

st.markdown("""
### Vanessa Santoso
**NIM : 2702242171**

Deep Learning Project - Long Short Term Memory (LSTM)
""")

st.divider()

# =================================
# INPUT
# =================================
text = st.text_area(
    "Masukkan Review",
    height=200,
    placeholder="Contoh: Produk sangat bagus dan pengiriman cepat"
)

# =================================
# PREDICTION
# =================================
if st.button("Predict Sentiment"):

    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu")
    else:

        seq = tokenizer.texts_to_sequences([text])

        padded = pad_sequences(
            seq,
            maxlen=max_len,
            padding='post'
        )

        score = model.predict(padded)[0][0]

        if score >= 0.5:

            st.success("😊 Sentimen Positif")

            st.metric(
                "Confidence",
                f"{score*100:.2f}%"
            )

        else:

            st.error("😡 Sentimen Negatif")

            st.metric(
                "Confidence",
                f"{(1-score)*100:.2f}%"
            )

st.divider()

st.write(
    "Created by Vanessa Santoso (2702242171)"
)