import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import pickle
from annoy import AnnoyIndex


feature_vectors = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Annoy index
feature_length = feature_vectors.shape[1]
annoy_index = AnnoyIndex(feature_length, 'euclidean')
for i, feature in enumerate(feature_vectors):
    annoy_index.add_item(i, feature)
annoy_index.build(10)

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
model.trainable = False

# GlobalMaxPooling layer to the model
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

def recommend(features, annoy_index, num_recommendations=5):
    indices = annoy_index.get_nns_by_vector(features, num_recommendations)
    return indices

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        with st.spinner('Processing...'):
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            indices = recommend(features, annoy_index)
        st.success('Done!')
        st.header("Recommendations")
        cols = st.columns(5)
        for col, idx in zip(cols, indices):
            with col:
                st.image(filenames[idx])
    else:
        st.header("Some error occurred in uploading")
