import streamlit as st
import joblib
import re
import string

# Load vectorizer and model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# Cleaning function
def filtering(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Streamlit app
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title('📰 Fake News Detector')
st.subheader('Check whether a news article is Real or Fake!')

# Two input fields
title_input = st.text_input('Enter News Title:')
text_input = st.text_area('Enter News Text:', height=200)

def prediction(input_text):
    cleaned_text = filtering(input_text)
    transformed_input = vectorizer.transform([cleaned_text])
    result = model.predict(transformed_input)
    return result[0]

if st.button('Predict'):
    
    if title_input and text_input:
        final_input = title_input + " " + text_input
    elif title_input:
        final_input = title_input
    elif text_input:
        final_input = text_input
    else:
        final_input = None
    
    if final_input:
        result = prediction(final_input)
        if result == 0:
            st.error('❌ The News is FAKE')
        else:
            st.success('✅ The News is REAL')
    else:
        st.warning('Please enter either title or text to check!')
