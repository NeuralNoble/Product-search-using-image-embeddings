import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

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


def extract_features(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.resnet50.preprocess_input(expanded_img_array)
    prediction = model.predict(preprocessed_img).flatten()
    normalized_prediction = prediction / norm(prediction)
    return normalized_prediction


filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

feature_vectors = []

for file in tqdm(filenames):
    feature_vectors.append(extract_features(file, model))

pickle.dump(feature_vectors,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
