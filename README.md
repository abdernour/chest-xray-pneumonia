# chest x-ray pneumonia detection

binary classification cnn that detects pneumonia from chest x-ray images (normal vs pneumonia). trained on the kaggle chest x-ray dataset using pytorch.

## dataset

- source: [kaggle chest x-ray images (pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 5,863 jpeg images across train/test/val splits
- 2 classes: normal, pneumonia

## model architecture

custom cnn with 4 convolutional blocks:
- conv1: 3 -> 64 filters (2x conv + bn + relu + maxpool + dropout)
- conv2: 64 -> 128 filters
- conv3: 128 -> 256 filters
- conv4: 256 -> 512 filters
- adaptive average pooling
- fully connected: 512 -> 256 -> 128 -> 2
- total parameters: ~6.7M

## preprocessing

medical-specific pipeline applied to each x-ray:
1. intensity normalization (min-max scaling)
2. clahe contrast enhancement (clip limit 2.0, tile grid 8x8)
3. bilateral denoising (d=9, sigma=75)

grayscale images are converted to 3-channel rgb for imagenet-normalized input.

## training

- optimizer: adam (lr=0.0001, weight decay=1e-4)
- loss: cross-entropy with inverse-frequency class weights (handles imbalanced dataset)
- scheduler: reduce on plateau (patience=5, factor=0.5)
- early stopping: patience=8 epochs
- augmentation: random rotation (10 deg), affine translation, horizontal flip

## evaluation metrics

standard accuracy plus medical-specific metrics:
- sensitivity (pneumonia recall) — how well it catches pneumonia cases
- specificity (normal recall) — how well it identifies healthy patients
- positive/negative predictive value
- roc-auc score
- confusion matrix

## results

training curves, confusion matrix, and roc curve are saved in the `results/` directory after training.

## project structure

```
pneumonia_cnn.py         - cnn model, dataset loader, preprocessing
train_pneumonia.py       - training loop, evaluation, visualization
pneumonia_example.py     - full training pipeline (entry point)
gradio_app.py            - interactive web application demo
requirements.txt         - python dependencies
results/                 - training output plots
```

## usage

```bash
# install dependencies
pip install -r requirements.txt

# download the dataset from kaggle and extract to archive/chest_xray/

# train (full 20 epochs)
python pneumonia_example.py

# train (quick demo, 10 epochs)
python pneumonia_example.py demo

# test on random samples
python test_pneumonia.py

# predict on a single x-ray
python test_pneumonia.py path/to/xray.jpg

# launch interactive web app
python gradio_app.py
```

## requirements

- python 3.8+
- pytorch
- opencv-python
- numpy, matplotlib, scikit-learn, seaborn, tqdm, pillow
