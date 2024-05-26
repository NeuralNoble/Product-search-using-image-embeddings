import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import pickle
from annoy import AnnoyIndex
import pandas as pd
import json


@st.cache_resource
def load_product_info(directory_path='styles'):
    product_info = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                product_id = data['data']['id']
                product_info[product_id] = data['data']
    return product_info


product_info = load_product_info()


@st.cache_resource
def load_embeddings_and_filenames():
    feature_vectors = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    return feature_vectors, filenames


feature_vectors, filenames = load_embeddings_and_filenames()


@st.cache_resource
def build_annoy_index(feature_vectors):
    feature_length = feature_vectors.shape[1]
    annoy_index = AnnoyIndex(feature_length, 'euclidean')
    for i, feature in enumerate(feature_vectors):
        annoy_index.add_item(i, feature)
    annoy_index.build(50)  # You can increase the number of trees for better accuracy
    return annoy_index


annoy_index = build_annoy_index(feature_vectors)

# Load the pre-trained ResNet50 model
@st.cache_resource
def load_model():
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
    return model

model = load_model()




def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error: {e}")
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
    recommended_products = [(idx, filenames[idx]) for idx in indices]
    return recommended_products


st.title("Fashion Recommender System")
st.markdown("Upload an image to get fashion recommendations based on the uploaded image.")

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image')

        with st.spinner('Processing...'):
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            recommendations = recommend(features, annoy_index)

        st.success('Done!')

        st.header("Recommendations")
        cols = st.columns(5)
        for col, (idx, filename) in zip(cols, recommendations):
            with col:
                product_id_str = os.path.splitext(os.path.basename(filename))[0]
                product_id = int(product_id_str)
                product_name = product_info.get(product_id, {}).get('productDisplayName', 'No Name')
                st.image(filename, use_column_width=True)
                st.caption(product_name)


    else:
        st.error("Some error occurred in uploading the image.")

st.markdown(
    """
    <style>
    .css-1aumxhk {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
