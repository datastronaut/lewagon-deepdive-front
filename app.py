import streamlit as st
from PIL import Image
import librosa
import requests

'''
# Le Wagon - Deep Dive project
'''

# ouvrir une image à partir d'une url:
# url = 'https://media.fisheries.noaa.gov/styles/original/s3/dam-migration/640x427-minke-whale.png'
# response = requests.get(url, stream=True)
# response.raw.decode_content = True
# image = Image.open(response.raw)

image = Image.open('images/cover.png')

st.image(image, caption='cover', use_column_width=False, width = 500 )


uploaded_file = st.file_uploader('Load you file here',type=['wav'])

st.audio(uploaded_file, format="audio/wav", start_time=0)

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file)

    st.pyplot(librosa.display.waveshow(y))
