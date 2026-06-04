import streamlit as st
import pickle

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🤖"
)

@st.cache_resource
def load_model():

    model = pickle.load(
        open("model/model.pkl", "rb")
    )

    vectorizer = pickle.load(
        open("model/vectorizer.pkl", "rb")
    )

    return model, vectorizer

model, vectorizer = load_model()

st.title("🤖 Sentiment Analysis")

st.markdown("""
### Vanessa Santoso
**NIM: 2702242171**
""")

text = st.text_area(
    "Masukkan Review"
)

if st.button("Predict"):

    data = vectorizer.transform([text])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data).max()

    if prediction == 1:
        st.success("😊 Positif")
    else:
        st.error("😡 Negatif")

    st.metric(
        "Confidence",
        f"{probability*100:.2f}%"
    )