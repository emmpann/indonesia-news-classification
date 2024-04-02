import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

labels = ['finance', 'food', 'hot', 'inet', 'oto', 'sport', 'travel']
model = load_model('model\model.hdf5')
with open('model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

# Fungsi untuk melakukan klasifikasi teks menggunakan model
def classify_text(text, model):
    
    text_seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=16)

    model_predict = model.predict([text_seq])

    prediction = [labels[np.argmax(i)] for i in model_predict]
    
    # bar chart
    model_predict = np.array(model_predict).flatten()
    data_array = np.array(model_predict)
    df = pd.DataFrame(data_array, index=labels)

    st.bar_chart(df, color=["#42b6f5"])

    return prediction[0], max(model_predict)

st.set_page_config(page_title="Klasifikasi Berita")

# Tampilan web menggunakan Streamlit
st.title("Klasifikasi Berita")

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks berita:")

if st.button("Classify"):
    if user_input.strip() != "":
        # Melakukan klasifikasi teks dengan model
        prediction = classify_text(user_input, model)

        st.success(f"Hasil Klasifikasi: {prediction[0]}")
        st.success(f"Confidence: {int(prediction[1]*100)}%")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")