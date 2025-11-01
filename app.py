
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# تحميل الموديل
model = load_model("my_model.h5")

# تحميل tokenizer و id2tag
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("id2tag.pkl", "rb") as f:
    id2tag = pickle.load(f)

st.title("🧠 Named Entity Recognition App (Custom Model)")
st.write("اكتب جملة والموديل هيطلعلك نوع كل كلمة 👇")

# إدخال الجملة
sentence = st.text_input("✍️ Sentence:")

if st.button("Analyze"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        # تقسيم الجملة لكلمات
        words = sentence.split()

        # تحويل الكلمات لتسلسل أرقام
        X = tokenizer.texts_to_sequences([words])
        X = pad_sequences(X, maxlen=model.input_shape[1], padding='post')

        # تنبؤ الموديل
        pred = model.predict(X)
        pred_ids = np.argmax(pred, axis=-1)[0]

        # تحويل الأرقام إلى تسميات
        decoded_tags = [id2tag[i] for i in pred_ids[:len(words)]]

        # عرض النتائج
        st.subheader("📋 Word → Tag")
        for w, t in zip(words, decoded_tags):
            st.write(f"{w} → {t}")
