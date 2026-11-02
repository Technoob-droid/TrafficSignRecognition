import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dataset import load_training_data
from model import build_cnn

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "traffic_sign_cnn.keras")

    print("[INFO] Loading data...")
    X_train, X_val, y_train, y_val = load_training_data(data_dir)
    print(f"[INFO] Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        fill_mode="nearest"
    )
    datagen.fit(X_train)

    model = build_cnn(input_shape=(32, 32, 3), num_classes=43)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True)
    ]

    print("[INFO] Training...")
    model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=30, validation_data=(X_val, y_val), callbacks=callbacks)
    print(f"[SUCCESS] Saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
