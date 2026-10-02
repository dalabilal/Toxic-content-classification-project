"""Streamlit app: classify user text or image captions and save everything in SQLite.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from PIL import Image

import database as db
from classifier import predict
from imagecaption import generate_caption

st.set_page_config(page_title="Toxic Content Classifier", layout="wide")
db.init_db()

st.title("Toxic Content Classification")
page = st.sidebar.radio("Choose input", ["Text", "Image", "View database"])


def show_result(label, confidence, probs):
    st.success(f"Predicted category: **{label}** (confidence {confidence:.1%})")
    st.bar_chart(pd.Series(probs).sort_values(ascending=False))


if page == "Text":
    text = st.text_area("Enter text", height=150)
    if st.button("Classify text"):
        if not text.strip():
            st.warning("Please type some text first.")
        else:
            label, conf, probs = predict(query=text.strip())
            db.log_entry("text", text.strip(), label, conf)
            show_result(label, conf, probs)

elif page == "Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if file is not None:
        image = Image.open(file)
        st.image(image, width=400)
        if st.button("Caption and classify image"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(image)
            st.write("**Generated caption:**", caption)
            label, conf, probs = predict(image_text=caption)
            db.log_entry("image", caption, label, conf)
            show_result(label, conf, probs)

else:
    st.subheader("All stored inputs and classifications")
    choice = st.selectbox("Show", ["all", "text", "image"])
    logs = db.get_logs(None if choice == "all" else choice)
    st.caption(f"{len(logs)} record(s)")
    st.dataframe(logs, use_container_width=True, hide_index=True)
    st.download_button("Download as CSV", logs.to_csv(index=False).encode("utf-8"),
                       file_name="toxic_logs.csv", mime="text/csv")