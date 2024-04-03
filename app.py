import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import webbrowser

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


st.set_page_config(page_title="Aplikasi Klasifikasi Berita :newspaper:")

activities = ["About","Data", "Prediction"]
choice = st.sidebar.selectbox("Choose Activity", activities)

if choice=="Data":
    st.title('Dataset')
    st.write("Berikut DataFrame dari dataset `Detik.com` berjumlah 69.700 data.")
    data = pd.read_csv("detik_news.csv")
    st.write(data)
if choice=="About":
    with st.container():
        st.title("Selamat datang di Aplikasi Klasifikasi Berita Berbahasa Indonesia:wave:")
        st.markdown("![Web Application](https://i.gifer.com/991p.gif)")
        st.markdown(""" 
        #### Built with Streamlit
        ## By
        + Kiagus Muhammad Efan Fitriyan
        """)
        st.markdown("""+ Lous Garcia""")
        st.markdown("""+ Nadia Laras""")
        st.markdown("""+ Nadhifa Faiza""")
        st.markdown("""+ Tri Rahmadhini""")
        url = 'https://github.com/emmpann/indonesia-news-classification.git'
        if st.button('Github'):
            webbrowser.open_new_tab(url)
if choice=="Prediction":
     # Tampilan web menggunakan Streamlit
    st.title("Klasifikasi Berita:newspaper:")
    st.header('mengklasifikasikan berita berdasarkan kategori tertentu')

    # Input teks dari pengguna
    user_input = st.text_area("Masukkan teks berita:", "Ketik disini", max_chars=100)

    if st.button("Classify"):
        if user_input.strip() != "":
            # Melakukan klasifikasi teks dengan model
            prediction = classify_text(user_input, model)

            st.success(f"Hasil Klasifikasi: {prediction[0]}")
            st.success(f"Confidence: {int(prediction[1]*100)}%")
        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")
     

# st.sidebar.subheader('About the App')
# st.sidebar.write('Text Classification App with Streamlit using a trained Naive Bayes model')
# st.sidebar.write("This is just a small text classification app. Don't fret if the prediction is not correct or if it is not what you expected, the model is not perfect.")
# st.sidebar.write("There is no provision for neutral text, yet…")