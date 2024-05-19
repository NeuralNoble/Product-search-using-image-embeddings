import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import os
import pickle
from sklearn.neighbors import NearestNeighbors
import pandas as pd

feature_vectors = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])

st.title("Fashion Recommender System")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())

        return 1

    except:
        return 0


def feature_extraction(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.resnet50.preprocess_input(expanded_img_array)
    prediction = model.predict(preprocessed_img).flatten()
    normalized_prediction = prediction / norm(prediction)
    return normalized_prediction


def recommend(features, feature_vectors):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_vectors)
    distances, indices = neighbors.kneighbors([features])
    return indices


uploaded_file = st.file_uploader("choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        with st.spinner('Processing...'):
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            indices = recommend(features, feature_vectors)
        st.success('Done!')
        st.header("Recommendations")
        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:
            st.image(filenames[indices[0][0]])

        with col2:
            st.image(filenames[indices[0][1]])

        with col3:
            st.image(filenames[indices[0][2]])

        with col4:
            st.image(filenames[indices[0][3]])

        with col5:
            st.image(filenames[indices[0][4]])

    else:
        st.header("Some Error Ocurred in uploading")
