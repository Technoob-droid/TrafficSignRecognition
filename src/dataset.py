import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMG_SIZE = (32, 32)
NUM_CLASSES = 43

def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_eq.astype("float32") / 255.0

def load_training_data(data_dir):
    train_dir = os.path.join(data_dir, "Train")
    images = []
    labels = []

    for class_id in range(NUM_CLASSES):
        class_folder = os.path.join(train_dir, str(class_id))
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('.png', '.jpg', '.ppm')):
                img_path = os.path.join(class_folder, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(preprocess_image(img))
                    labels.append(class_id)

    X = np.array(images, dtype="float32")
    y = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
