import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF info and warnings

import argparse, numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

LABELS = ["airplane","automobile","bird","cat","deer",
          "dog","frog","horse","ship","truck"]

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((32, 32))
    x = np.asarray(img, dtype="float32") / 255.0
    return x[None, ...]  # (1,32,32,3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to JPG/PNG")
    ap.add_argument("--model", default="models/cifar10_cnn.h5")
    args = ap.parse_args()

    model = load_model(args.model)
    x = preprocess(Image.open(args.image))
    probs = model.predict(x, verbose=0)[0]
    k = int(np.argmax(probs))
    print(f"Predicted: {LABELS[k]}  (p={probs[k]:.3f})")

if __name__ == "__main__":
    main()
