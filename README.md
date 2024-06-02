# Fashion Recommender System
This project is a fashion recommender system built using Streamlit, TensorFlow, and Keras. It allows users to upload an image of a fashion item and receive recommendations for similar items based on image features extracted using the ResNet50 model.

## How It Works
1.  Users can upload an image of a fashion item through the Streamlit interface. The uploaded image is saved in the uploads directory and displayed on the web page.

2. The uploaded image is processed and resized to 224x224 pixels, the required input size for the ResNet50 model. The pre-trained ResNet50 model, modified with a GlobalMaxPooling2D layer, extracts a feature vector from the image. This feature vector is normalized for consistency.

3.  The extracted feature vector is compared to a set of precomputed feature vectors (stored in embeddings.pkl) using a Nearest Neighbors algorithm. The algorithm finds the most similar items by calculating the Euclidean distances between the feature vectors.

4. The system retrieves and displays the top 5 most similar fashion items based on the similarity of the feature vectors. The images of these recommended items are shown side-by-side on the Streamlit interface.

## Demo
Check out the demo video to see the fashion recommender system in action!

[![Demo Video](https://img.youtube.com/vi/pAX5YljK4Vk/0.jpg)](https://www.youtube.com/watch?v=pAX5YljK4Vk)

<img width="500px" alt="Screenshot 2024-05-26 at 1 42 32 PM" src="https://github.com/NeuralNoble/Fashion-Recommender-System/assets/156664113/ca438eae-d660-4b25-99f0-2a887fff392f">
