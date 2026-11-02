from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn(input_shape=(32, 32, 3), num_classes=43):
    return Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
