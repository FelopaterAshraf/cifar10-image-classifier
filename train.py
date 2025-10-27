import os, argparse, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models

LABELS = ["airplane","automobile","bird","cat","deer",
          "dog","frog","horse","ship","truck"]

def build_model(input_shape=(32,32,3), num_classes=10): 
    m= models.Sequential([
    layers.Conv2D(32,(3,3), input_shape = (32,32,3), padding = 'same', activation = 'relu'), 
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2), strides=2),
    layers.Dropout(0.3),
    #Block
    layers.Conv2D(64,(3,3), padding = 'same', activation = 'relu', strides=1),
    layers.BatchNormalization(),
    layers.Conv2D(64,(3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2), strides=2),
    layers.Dropout(0.3),

    #Block
    layers.Conv2D(128,(3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128,(3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2), strides=2),
    layers.Dropout(0.3),

    #Final Block
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(10, activation = 'softmax')
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def main(epochs=10, batch_size=128, out_path="models/cifar10_cnn.h5"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
    y_tr = y_tr.squeeze().astype("int64")
    y_te = y_te.squeeze().astype("int64")
    x_tr = (x_tr.astype("float32")/255.0)
    x_te = (x_te.astype("float32")/255.0)

    model = build_model()
    hist = model.fit(
        x_tr, y_tr,
        validation_data=(x_te, y_te),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    model.save(out_path)
    np.save("models/history.npy", hist.history, allow_pickle=True)
    print(f"Saved model -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default="models/cifar10_cnn.h5")
    a = ap.parse_args()
    main(epochs=a.epochs, batch_size=a.batch_size, out_path=a.out)
