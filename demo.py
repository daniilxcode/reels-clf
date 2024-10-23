import streamlit as st
from pytubefix import YouTube
from reels_clf import ReelsClassifier

clf = ReelsClassifier()

url = st.text_input('Ссылка на видео')
try:
    yt = YouTube(url)
    author = yt.author
    title = yt.title
    description = yt.description
except:
    author = None
    title = None
    description = None

with st.form('reels-clf'):
    channel_name = st.text_input('Название канала', author)
    reel_name = st.text_input('Название ролика', title)
    description = st.text_area('Описание ролика', description)
    submit = st.form_submit_button('Предсказать')

if submit:
    inputs = {'channel_name': channel_name, 'reel_name': reel_name, 'description': description}
    predictions = clf.predict_proba(inputs)
    st.write(f'Предсказание: **{predictions[0]["label"]}**')
    st.bar_chart(predictions, x='label', y='score', horizontal=True)
