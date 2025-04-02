import pickle

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

flower_type= None

model = load_model("E:\pythonds\skillDatascience\Machine learning\Deployment\iris_class_tf.keras")

with open("E:\pythonds\skillDatascience\Machine learning\Deployment\iris_processor.pkl", "rb") as file:
    preprocessor = pickle.load(file)


def predict_flower(user_input: list):
    scaled_data = preprocessor['scaler'].transform(user_input)
    
    y_pred = tf.one_hot(
        np.argmax(model.predict(scaled_data), axis =1),
        depth = 3
    ).numpy()

    return preprocessor['encoder'].inverse_transform(y_pred)[0][0]


if __name__ == "__main__":
    st.title("Flower Classification by _AI_")

    petal_length = st.number_input(
        "Enter petal length of your Flower", min_value = 0.00,
    )

    petal_width = st.number_input(
        "Enter petal width of your Flower", min_value = 0.00,
    )

    sepal_length = st.number_input(
        "Enter sepal length of your Flower", min_value = 0.00,
    )

    sepal_width = st.number_input(
        "Enter sepal width of your Flower", min_value = 0.00,
    )
    
    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]


    btn = st.button("Submit")
    if btn:
        
         flower_type = predict_flower(user_input)
         print(flower_type)

         st.markdown(f" Flower is :**{flower_type}**")
