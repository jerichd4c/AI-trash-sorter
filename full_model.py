import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras import layers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2, EfficientNetB0
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# config

IMG_SIZE= (224, 224)
BATCH_SIZE= 8
EPOCHS= 30
NUM_CLASSES= 12 
DATASET_PATH= 'dataset'

# data augmentation and preprocessing

def load_and_preprocess_data(dataset_path):

    # verify if theres enough data for the training

    classes = os.listdir(dataset_path)
    print(f"Clases encontradas: {classes}")

    total_images = 0 

    for class_name in classes: 
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Clase '{class_name}': {num_images} imágenes")
            total_images += num_images

    print(f"Total de imágenes encontradas: {total_images}")
          
    if total_images < NUM_CLASSES * 2:
            print("Advertencia: No hay suficientes datos para entrenar el modelo. Se recomienda al menos 2 imágenes por clase.")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2,
    )

    # load training data

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode='rgb',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb',
        shuffle=True
    )

    return train_generator, validation_generator

# create model

def create_base_model(num_classes):
    # pre trained model 
    9
    base_model = MobileNetV2( 
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

    print("Cargando datos...")
    train_gen, val_gen = load_and_preprocess_data(DATASET_PATH)

    # if theres no data, use all data for training (debug)
    if val_gen.samples == 0:
        print("No se encontraron datos de validación. Usando todos los datos para entrenamiento.")
        train_gen, val_gen = load_and_preprocess_data(DATASET_PATH)
        # deactivate validation
        val_gen = None

    print("Compilando modelo...")
    model = create_base_model(NUM_CLASSES)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [ 
        keras.callbacks.EarlyStopping(monitor='val_accuracy' if val_gen else 'accuracy', patience=8, restore_best_weights=True),   
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy' if val_gen else 'loss', factor=0.2, patience=5, min_lr=1e-7)
    ]
    
    # first training

    print("Iniciando entrenamiento...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # fine tuning only IF theres enough data

    if train_gen.samples > 50:
        print("Iniciando fine-tuning...")
        base_model = model.layers[0]
        base_model.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history_fine = model.fit(
            train_gen,
            epochs=EPOCHS+10,
            initial_epoch=history.epoch[-1],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )   
    else :
        history_fine = None 
        print("No hay suficientes datos para fine-tuning.")

    return model, history, history_fine, train_gen, val_gen

# evaluate model

def evaluate_model(model, val_gen, train_gen=None): 
    if val_gen is None or val_gen.samples == 0:
        print("No hay datos de validación disponibles para evaluar el modelo.")
        return None, None, None
    
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
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return y_true, y_pred, class_names

# main function (generates the h5 model file)

if __name__ == "__main__":

    # verify if DATASET_PATH is correctly set
    if not os.path.exists(DATASET_PATH):
        print(f"Error: No se encuentra el directorio '{DATASET_PATH}")
        print("Estructura de carpetas: dataset/clase1/ , dataset/clase2/ , ...")
    else:
        try:
            # train the model
            model, history, history_fine, train_gen, val_gen = train_model()

            # evaluate the model (only if the validation set exists)
            if val_gen and val_gen.samples > 0:
                y_true, y_pred, class_names = evaluate_model(model, val_gen)
            else:
                print("No se evaluó el modelo debido a la falta de datos de validación.")

            # save the model
            model.save('waste_classifier_model.h5')
            print("Modelo guardado como 'waster_classifier_model.h5'")

            # training graphs

            plt.figure(figsize=(12, 4))

            # plot accuracy

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Entrenamiento')
            if val_gen and val_gen.samples > 0:
                plt.plot(history.history['val_accuracy'], label='Validación')
            if history_fine:
                plt.plot(history_fine.history['accuracy'], label='Entrenamiento Fine-tuning')
                if val_gen and val_gen.samples > 0:
                    plt.plot(history_fine.history['val_accuracy'], label='Validación Fine-tuning')

            plt.title('Precisión del modelo')
            plt.xlabel('Época')
            plt.ylabel('Precisión')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # plot loss

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Entrenamiento')
            if val_gen and val_gen.samples > 0:
                plt.plot(history.history['val_loss'], label='Validación')
            if history_fine:
                plt.plot(history_fine.history['loss'], label='Entrenamiento Fine-tuning')
                if val_gen and val_gen.samples > 0:
                    plt.plot(history_fine.history['val_loss'], label='Validación Fine-tuning')

            plt.title('Pérdida del modelo')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.show()

            print ("Entrenamiento completado.")

        except Exception as e:
            print(f"Ocurrió un error durante el entrenamiento: {e}")
            print("Cosas a verificar:")
            print("1. Aumenta el numero de imágenes en el dataset.")
            print("2. Verifica la estructura de carpetas del dataset.")
            print("3. Verifica el formato de las imágenes (jpg, png, jpeg).")