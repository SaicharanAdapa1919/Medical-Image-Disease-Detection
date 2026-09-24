import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import build_model

train_dir = "dataset/train"
test_dir = "dataset/test"
model_path = "model.h5"
class_map_path = "class_indices.json"

img_size = (224, 224)
batch_size = 32
epochs = 15

def train_model():
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    # Only rescaling for test/validation
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Save class indices for app.py to use
    class_indices = {v: k for k, v in train_gen.class_indices.items()}
    with open(class_map_path, 'w') as f:
        json.dump(class_indices, f)
    print(f"Class mapping saved to {class_map_path}: {class_indices}")

    # Build model
    model = build_model(num_classes=train_gen.num_classes)
    print(model.summary())

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    # Train
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # If best model was saved by checkpoint, load it; otherwise save current
    if os.path.exists(model_path):
        print(f"Best model loaded from {model_path}")
    else:
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # Evaluate on test set
    loss, accuracy = model.evaluate(test_gen)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Loss: {loss:.4f}")

    return model, history

if __name__ == "__main__":
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print("ERROR: Training data not found. Please run generate_sample_data.py first.")
        exit(1)
    train_model()
