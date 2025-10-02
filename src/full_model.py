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
import kagglehub
import shutil

# config

IMG_SIZE= (224, 224)
BATCH_SIZE= 16
EPOCHS= 25
NUM_CLASSES= 12 
KAGGLE_DATASET= "mostafaabla/garbage-classification"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATASET_PATH= os.path.join(BASE_DIR, 'dataset') 
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'waste_classifier_model.h5')

# creat dir if not exists

os.makedirs(RESULTS_DIR, exist_ok=True)

# download dataset from kaggle if not exists

def setup_dataset():

    if os.path.exists(DATASET_PATH):
        print(f"El directorio '{DATASET_PATH}' ya existe. Usando el dataset existente.")
        return DATASET_PATH 
    
    print("Descargando dataset desde Kaggle...")
    try:

        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"Dataset descargado en: {download_path}")

        actual_dataset_path = find_image_folder(download_path)

        if actual_dataset_path: 
            print(f"Se encontro el dataset en: {actual_dataset_path}")

            # creates local dirs
            os.makedirs(DATASET_PATH, exist_ok=True)

            copy_classes_to_dataset(actual_dataset_path, DATASET_PATH)  
            return DATASET_PATH
        else:
            print("No se pudo encontrar el dataset dentro del archivo descargado.")
            return None

    except Exception as e:
        print(f"Ocurrió un error al descargar el dataset: {e}")
        return None

# finds the folder with recursive search

def find_image_folder(base_path):

    expected_paths = [
        "Garbage classification",
        "Garbage Classification",
        "garbage classification",
        "garbage",
        "dataset",
        "images",
        "train",
        "waste"
    ]

    for root, dirs, files in os.walk(base_path):
       
       if len(dirs) >5:
         
       # verify the tags 

            class_like_dirs = [d for d in dirs if any(keyword in d.lower() for keyword in 
                                                      ['glass', 'paper', 'plastic', 'metal', 'trash', 'cardboard', 'biological', 'battery', 'clothes', 'shoes', 'electronics', 'organic'])]
            if len(class_like_dirs) >= 5:
                return root
       
            for possible_dir in expected_paths:
                if possible_dir in dirs:
                    return os.path.join(root, possible_dir)
    
    for root, dirs, files in os.walk(base_path):
        if len(dirs) >= NUM_CLASSES:
            return root
        
    return None

# copy the classes to the dataset folder

def copy_classes_to_dataset(source_path, dest_path):

    # get all subdirs

    subdirs = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]

    print (f"Clases encontradas: {subdirs}")

    for subdir in subdirs:
        src_dir = os.path.join(source_path, subdir)
        dst_dir = os.path.join(dest_path, subdir)

        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] 

        if image_files: 
            print (f"Copiando {len(image_files)} imágenes desde '{subdir}'...")
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        else:
            print (f"No se encontraron imágenes en la clase '{subdir}', no  contiene imagenes.")

# data augmentation and preprocessing

def load_and_preprocess_data(dataset_path):

    # verify dataset structure

    if not os.path.exists(dataset_path):
        print(f"El directorio '{dataset_path}' no existe.")
        return None, None

    classes = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    if len(classes) == 0:
        print("No se encontraron clases en el dataset. Verifica la estructura de carpetas.")
        return None, None

    print(f"Número de clases encontradas: {len(classes)}")
    print(f"Clases: {classes}")

    # verify if theres enough data for the training

    total_images = 0 

    for class_name in classes: 
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Clase '{class_name}': {num_images} imágenes")
            total_images += num_images

    print(f"Total de imágenes encontradas: {total_images}")
          
    if total_images < NUM_CLASSES * 5:
            print("Advertencia: No hay suficientes datos para entrenar el modelo. Se recomienda al menos 5 imágenes por clase.")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.4,
        shear_range=0.3,
        brightness_range=(0.7, 1.3),
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
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    print("Modelo creado con MobileNetV2 como base.")
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
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [ 
        keras.callbacks.EarlyStopping(monitor='val_accuracy' if val_gen else 'accuracy', patience=8, restore_best_weights=True),   
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy' if val_gen else 'loss', factor=0.2, patience=5, min_lr=1e-7)
    ]
    

    print("Iniciando entrenamiento...")

    # train without validation

    if val_gen is None or val_gen.samples == 0:
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        history_fine = None

    else:


    # first training

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
            optimizer=Adam(learning_rate=0.0001),
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

    filepath = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Matriz de confusión guardada en: {filepath}")
    plt.show()

    return y_true, y_pred, class_names

# main function (generates the h5 model file)

if __name__ == "__main__":
    print ("Configurando dataset...")
    dataset_path = setup_dataset()

    # verify if DATASET_PATH is correctly set

    if dataset_path is None:
        print("No se pudo configurar el dataset.") 
        print("Estructura de carpetas: dataset/clase1/ , dataset/clase2/ , ...")
        exit(1)

    try:
            # train the model
            print ("Iniciando el entrenamiento del modelo...")
            model, history, history_fine, train_gen, val_gen = train_model()

            if model is None:
                print("El entrenamiento del modelo falló.")
                exit(1)

            # evaluate the model (only if the validation set exists)
            if val_gen and val_gen.samples > 0:
                y_true, y_pred, class_names = evaluate_model(model, val_gen)
            else:
                print("No se evaluó el modelo debido a la falta de datos de validación.")

            # save the model
            model.save(MODE_SAVE_PATH)
            print(f"Modelo guardado como: {MODE_SAVE_PATH}")

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
            filepath = os.path.join(RESULTS_DIR, 'training_history.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Gráficas de entrenamiento guardadas en: {filepath}")

            plt.show()

            print ("Entrenamiento completado.")

    except Exception as e:
            print(f"Ocurrió un error durante el entrenamiento: {e}")
            print("Cosas a verificar:")
            print("1. Aumenta el numero de imágenes en el dataset.")
            print("2. Verifica la estructura de carpetas del dataset.")
            print("3. Verifica el formato de las imágenes (jpg, png, jpeg).")