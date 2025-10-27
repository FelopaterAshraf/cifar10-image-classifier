# app.py
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

LABELS = ["airplane","automobile","bird","cat","deer",
          "dog","frog","horse","ship","truck"]

# Streamlit Page Setup
st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="📸", layout="wide")

@st.cache_resource
def load_cifar_model():
    return load_model("models/cifar10_cnn.h5")

model = load_cifar_model()
# Page Header
st.title("CIFAR-10 Image Classifier")
st.caption("Upload a JPG/PNG. The model resizes to 32×32 and predicts one of 10 classes.")

# --- Class Reference Section ---
st.subheader("CIFAR-10 Classes")


example_images = {
    "airplane": "GUI/airplane.jpg",
    "automobile": "GUI/car.jpg",
    "bird": "GUI/bird.jpg",
    "cat": "GUI/cat.jpg",
    "deer": "GUI/deer.jpg",
    "dog": "GUI/dog.jpg",
    "frog": "GUI/frog.jpg",
    "horse": "GUI/horse.jpg",
    "ship": "GUI/ship.jpg",
    "truck": "GUI/truck.jpg",
}

cols = st.columns(5)
for i, (label, url) in enumerate(example_images.items()):
    with cols[i % 5]:
        st.image(url, caption=label, use_column_width=True)



st.markdown(
    """
    **Upload an image from one of the above classes** (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck)  
    to let the model classify it!
    """
)

# File Uploader
uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

#Image Preprocessing Function
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((32, 32))
    x = np.asarray(img, dtype="float32") / 255.0
    return x[None, ...]  # (1,32,32,3)

if uploaded:
    img = Image.open(uploaded)
    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    k = int(np.argmax(probs))

    st.image(img, caption=f"Predicted: {LABELS[k]} (p={probs[k]:.3f})", width=256)
else:
    st.info("Choose an image to get a prediction.")
