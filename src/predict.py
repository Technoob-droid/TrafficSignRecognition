import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from dataset import preprocess_image

CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5t', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Vehicles > 3.5t prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End no passing veh > 3.5t'
}

def predict(image_path, model_path):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load {image_path}")
        return
    tensor = np.expand_dims(preprocess_image(img), axis=0)
    preds = model.predict(tensor)[0]
    idx = int(np.argmax(preds))
    print(f"\nResult: {CLASSES.get(idx, 'Unknown')} ({preds[idx] * 100:.2f}% confidence)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_file = os.path.join(os.path.dirname(__file__), "..", "models", "traffic_sign_cnn.keras")
        predict(sys.argv[1], model_file)
    else:
        print("Usage: python predict.py <path_to_image>")
