import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras import layers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# config

IMG_SIZE= (224, 224)
BATCH_SIZE= 32
EPOCHS= 20
NUM_CLASSES= 12 
DATASET_PATH= 'dataset'

# data augmentation and preprocessing

def load_and_preprocess_data(DATASET_PATH):

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )

    # load training data

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# create model

def create_base_model(NUM_CLASSES):
    # pre trained model 
    base_model = EfficientNetB0( 
        weights='imagenet',
        include_top=False,
        input_shape=(220, 220, 3)
    )

    base_model.trainable = False
    
    # custom layers 

    model = keras.Sequential([
        base_model, 
        layers.GlobalAveragePooling2D(),
        layers.Droput(0.2),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

# train model

def train_model():

    print("Loading and preprocessing data...")
    train_generator, validation_generator = load_and_preprocess_data(DATASET_PATH)

    print("Compiling model...")
    model = create_base_model(NUM_CLASSES)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [ 
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),   
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
    ]
    
    # first training

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # second training (fine-tuning)

    base_model = model.layers[0]
    base_model.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=0.0001/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_generator,
        epochs=10,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        callbacks=callbacks
    )

    return model, history, history_fine, train_generator, validation_generator

# evaluate model

def evaluate_model(model, validation_generator): 

    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes

    # metrics 
    
    class_names = list(validation_generator.class_indices.keys())

    print ("Confusion Matrix:")
    print (classification_report(y_true, y_pred, target_names=class_names))

    # confusion matrix

    plt.figure(figsize=(12,10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Prediccion')
    plt.ylabel('Verdadero')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    return y_true, y_pred, class_names

# main function (generates the h5 model file)

if __name__ == "__main__":

    # verify if DATASET_PATH is correctly set
    if not os.path.exists(DATASET_PATH):
        print(f"Error: No se encuentra el directorio '{DATASET_PATH}")
    else:
        # train the model
        model, history, history_fine, train_generator, validation_generator = train_model()

        # evaluate the model
        y_true, y_pred, class_names = evaluate_model(model, validation_generator)

        # save the model
        model.save('trash_sorter_model.h5')
        print("Modelo guardado como 'trash_sorter_model.h5'")

        # training graphs

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Validación')
        plt.title('Precisión del modelo')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], label='Validación')
        plt.title('Pérdida del modelo')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()