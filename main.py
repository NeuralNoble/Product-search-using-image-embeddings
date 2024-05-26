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

styles_df = pd.read_csv('styles.csv',on_bad_lines='warn')
feature_vectors = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Build Annoy index
feature_length = feature_vectors.shape[1]
annoy_index = AnnoyIndex(feature_length, 'euclidean')
for i, feature in enumerate(feature_vectors):
    annoy_index.add_item(i, feature)
annoy_index.build(50)  # You can increase the number of trees for better accuracy

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
model.trainable = False

# Add GlobalMaxPooling layer to the model
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])

st.set_page_config(page_title="Fashion Recommender System", layout="wide")




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
    return indices


st.title("Fashion Recommender System")
st.markdown("Upload an image to get fashion recommendations based on the uploaded image.")

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image')

        with st.spinner('Processing...'):
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            indices = recommend(features, annoy_index)

        st.success('Done!')

        st.header("Recommendations")
        cols = st.columns(5)
        for col, idx in zip(cols, indices):
            with col:
                gender = styles_df.iloc[idx]['gender']
                product_display_name = styles_df.iloc[idx]['productDisplayName']
                st.image(filenames[idx], use_column_width=True)
                st.caption(f"{product_display_name}")
                st.caption(f"{gender}")
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
