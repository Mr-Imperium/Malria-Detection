import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import imutils.paths as paths

# Configuration
ORIG_INPUT_DATASET = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/"
BASE_PATH = "/kaggle/working/dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "training")
VAL_PATH = os.path.join(BASE_PATH, "validation")
TEST_PATH = os.path.join(BASE_PATH, "testing")
MODEL_PATH = "/kaggle/working/malaria_detection_model"

# Training configurations
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
NUM_EPOCHS = 50
INIT_LR = 1e-1
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 2

class ResNet:
    @staticmethod
    def residual_module(x, K, stride, chanDim, red=False, reg=1e-4):
        """Residual module for ResNet architecture"""
        shortcut = x

        # First block
        x = layers.Conv2D(K, (3, 3), strides=stride, padding="same", kernel_regularizer=regularizers.l2(reg))(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Activation("relu")(x)

        # Second block
        x = layers.Conv2D(K, (3, 3), padding="same", kernel_regularizer=regularizers.l2(reg))(x)
        x = layers.BatchNormalization(axis=chanDim)(x)

        # Reduce spatial dimensions if needed
        if red:
            shortcut = layers.Conv2D(K, (1, 1), strides=stride, padding="same", 
                                     kernel_regularizer=regularizers.l2(reg))(shortcut)

        # Add shortcut connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)

        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=1e-4):
        """Build ResNet model"""
        # Input layer
        inputShape = (width, height, depth)
        chanDim = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1

        inputs = layers.Input(shape=inputShape)

        # Initial convolution
        x = layers.Conv2D(filters[0], (3, 3), padding="same", kernel_regularizer=regularizers.l2(reg))(inputs)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Build stages
        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red=True, reg=reg)

            for j in range(1, stages[i]):
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, reg=reg)

        # Classifier
        x = layers.AveragePooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, kernel_regularizer=regularizers.l2(reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(classes, activation="softmax")(x)

        model = models.Model(inputs, x)
        return model

def create_dataset_splits():
    """Create train, validation, and test splits"""
    # Get all image paths
    imagePaths = list(paths.list_images(ORIG_INPUT_DATASET))
    np.random.seed(42)
    np.random.shuffle(imagePaths)

    # Create base directories
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(VAL_PATH, exist_ok=True)
    os.makedirs(TEST_PATH, exist_ok=True)

    # Split paths
    trainPaths, testPaths = train_test_split(imagePaths, test_size=1-TRAIN_SPLIT, random_state=42)
    valPaths, trainPaths = train_test_split(trainPaths, test_size=VAL_SPLIT, random_state=42)

    # Copy files to respective directories
    for split_name, paths_list, base_output in [
        ("training", trainPaths, TRAIN_PATH),
        ("validation", valPaths, VAL_PATH),
        ("testing", testPaths, TEST_PATH)
    ]:
        print(f"[INFO] Building '{split_name}' split")
        for inputPath in paths_list:
            filename = os.path.basename(inputPath)
            label = os.path.basename(os.path.dirname(inputPath))
            labelPath = os.path.join(base_output, label)
            os.makedirs(labelPath, exist_ok=True)
            
            dest_path = os.path.join(labelPath, filename)
            tf.io.gfile.copy(inputPath, dest_path, overwrite=True)

def create_data_generators():
    """Create data generators for training, validation, and testing"""
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_generator = val_datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def train_model():
    """Train the ResNet model"""
    # Create dataset splits
    create_dataset_splits()

    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators()

    # Initialize model
    model = ResNet.build(
        width=IMAGE_SIZE[0], 
        height=IMAGE_SIZE[1], 
        depth=3, 
        classes=NUM_CLASSES, 
        stages=(3, 4, 6), 
        filters=(64, 128, 256, 512), 
        reg=0.0005
    )

    # Compile model
    opt = SGD(learning_rate=INIT_LR, momentum=0.9)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=opt, 
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(MODEL_PATH, 'best_model.keras'), 
            monitor='val_accuracy', 
            save_best_only=True
        )
    ]

    # Train model
    history = model.fit(
        train_generator, 
        epochs=NUM_EPOCHS, 
        validation_data=val_generator,
        callbacks=callbacks,
    )

    # Evaluate model
    print("[INFO] Evaluating network...")
    test_generator.reset()
    predictions = model.predict(test_generator, steps=len(test_generator))
    pred_labels = np.argmax(predictions, axis=1)
    
    print(classification_report(
        test_generator.classes, 
        pred_labels, 
        target_names=list(test_generator.class_indices.keys())
    ))

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/kaggle/working/training_plot.png')

    # Save entire model for deployment
    model.save(os.path.join(MODEL_PATH, 'malaria_detection_model.keras'))

def main():
    train_model()

if __name__ == "__main__":
    main()