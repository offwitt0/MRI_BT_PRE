import streamlit as st
import tensorflow as tf
from PIL import Image as pilimage
import numpy as np
import os
import cv2
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dense, Layer
import keras.backend as K
import tempfile

# Custom f1_score function
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# Custom SEBlock Layer
class SEBlock(Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        super(SEBlock, self).build(input_shape)
        channels = input_shape[-1]
        self.global_pooling = GlobalAveragePooling2D()
        self.dense1 = Dense(channels // self.ratio, activation='relu')
        self.dense2 = Dense(channels, activation='sigmoid')

    def call(self, inputs):
        se = self.global_pooling(inputs)
        se = K.expand_dims(K.expand_dims(se, axis=1), axis=1)
        se = self.dense1(se)
        se = self.dense2(se)
        return inputs * se

    def compute_output_shape(self, input_shape):
        return input_shape

# Load the model
model_path = "CNN&SE&LSTM.h5"
if os.path.exists(model_path):
    st.write(f"Model file found at: {model_path}")
    model = load_model(model_path, custom_objects={'f1_score': f1_score, 'SEBlock': SEBlock})
else:
    st.write(f"Model file not found at: {model_path}")
    
def makepredictions(img):
    img_d = img.resize((256, 256))  # Resize the image to match the model input shape
    if len(np.array(img_d).shape) < 4:
        rgb_img = pilimage.new("RGB", img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img = img_d

    rgb_img = np.array(rgb_img, dtype=np.float64)
    rgb_img = np.expand_dims(rgb_img, axis=0)  # Add batch dimension
    predictions = model.predict(rgb_img)
    a = int(np.argmax(predictions))

    if a == 0:
        a = "Glioma Tumor"
    elif a == 1:
        a = "Meningioma Tumor"
    elif a == 2:
        a = "No Tumor"
    else:
        a = "Pituitary Tumor"
    return a

def generate_grad_cam(img):
    x = np.array(img.resize((256, 256)), dtype=np.float64)
    x = np.expand_dims(x, axis=0) / 255.0  # Normalize the pixel values

    last_conv_layer = model.get_layer('conv2d_34')

    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, np.argmax(predictions)]

    grads = tape.gradient(loss, conv_outputs)[0]

    cam = np.mean(conv_outputs[0], axis=-1)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))  # Resize CAM to match the size of the original image
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    x_uint8 = (x[0] * 255).astype(np.uint8)

    superimposed_img = cv2.addWeighted(
        x_uint8, 0.6,
        cv2.resize(heatmap, (256, 256)), 0.4, 0
    )

    return superimposed_img, heatmap

# Streamlit app
st.title("Tumor Detection and Grad-CAM Visualization")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        img = pilimage.open(temp_file.name)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = makepredictions(img)
        st.write(f"Prediction: {prediction}")

        superimposed_img, heatmap = generate_grad_cam(img)

        st.image(superimposed_img, caption='Superimposed Image.', use_column_width=True)
        st.image(heatmap, caption='Grad-CAM Heatmap.', use_column_width=True)
