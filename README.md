# CIFAR-10 Image Classifier (Streamlit + TensorFlow)
have a look: https://cifar10-image-classifier-dnaczpylopgc5s64fthwqn.streamlit.app/


Train a CNN on CIFAR-10, test on images, and run a Streamlit web app.

## Setup (local)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
## Train
python train.py --epochs 10

## Test a single image
python test.py [image name]

## Run the web app
streamlit run app.py

<img width="1795" height="940" alt="image" src="https://github.com/user-attachments/assets/82c855a3-7510-4528-8aae-9f347302244d" />
