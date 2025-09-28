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

# config

IMG_SIZE= (224, 224)
BATCH_SIZE= 8
EPOCHS= 20
NUM_CLASSES= 12 
DATASET_PATH= 'dataset'

# data augmentation and preprocessing

def load_and_preprocess_data(dataset_path):

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2,
    )

    # debug
    test_datagen = ImageDataGenerator(rescale=1./255)

    # load training data

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode='rgb' 
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb' 
    )

    return train_generator, validation_generator

# create model

def create_base_model(num_classes):
    # pre trained model 
    
    base_model = EfficientNetB0( 
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False
    
    # custom layers 

    model = keras.Sequential([
        base_model, 
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# train model

def train_model():

    print("Loading and preprocessing data...")
    train_gen, val_gen = load_and_preprocess_data(DATASET_PATH)

    print("Compiling model...")
    model = create_base_model(NUM_CLASSES)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [ 
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),   
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    # first training

    print("Iniciando entrenamiento...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
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

    # adjust epochs for fine-tuning
    fine_tune_epochs = 10
    total_epochs = len(history.epoch) + fine_tune_epochs

    history_fine = model.fit(
        train_gen,
        epochs=total_epochs,
        initial_epoch=len(history.epoch),
        validation_data=val_gen,
        callbacks=callbacks
    )

    return model, history, history_fine, train_gen, val_gen

# evaluate model

def evaluate_model(model, val_gen): 

    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes

    # metrics 
    
    class_names = list(val_gen.class_indices.keys())

    print ("Reporte de Clasificación:")
    print (classification_report(y_true, y_pred, target_names=class_names))

    # confusion matrix

    plt.figure(figsize=(12,10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
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
        model, history, history_fine, train_gen, val_gen = train_model()

        # evaluate the model
        y_true, y_pred, class_names = evaluate_model(model, val_gen)

        # save the model
        model.save('trash_sorter_model.h5')
        print("Modelo guardado como 'trash_sorter_model.h5'")

        # training graphs

        plt.figure(figsize=(12, 4))
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