import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0   
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import shutil

# config

IMG_SIZE= (224, 224)
BATCH_SIZE= 16
EPOCHS= 30
KAGGLE_DATASET= "feyzazkefe/trashnet"
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

    class_patterns = [
        'glass', 'paper', 'plastic', 'metal', 'trash', 
        'cardboard', 'biological', 'battery', 'clothes', 
        'shoes', 'electronics', 'organic'
    ]

    for root, dirs, files in os.walk(base_path):
         
       # verify the tags 

        class_dirs = [d for d in dirs if any(pattern in d.lower() for pattern in class_patterns)]

        if len(class_dirs) >= 5:
                has_images = True
                for class_dir in class_dirs:
                    class_path = os.path.join(root, class_dir)
                    if not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(class_path)):
                        has_images = False
                        break
                if has_images:
                    return root
    
    # Búsqueda alternativa
    for root, dirs, files in os.walk(base_path):
        if len(dirs) >= 5:
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

# verify dataset

def verify_dataset_integrity(dataset_path):

    # verify the dataset is correctly structured

    print ("Verificando dataset")
    if not os.path.exists(dataset_path):
        print(f"El directorio '{dataset_path}' no existe.")
        return False

    classes = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    if len(classes) < 2:
        print("Se necesitan al menos 2 clases para entrenar.")
        return False

    print(f"Número de clases encontradas: {classes}")
    total_images = 0

    for class_name in classes: 
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Clase '{class_name}': {len(images)} imágenes")
        
        if images: 
            sample_image = os.path.join(class_path, images[0])  
            try: 
                img = tf.keras.preprocessing.image.load_img(sample_image)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                print(f"Imagen de muestra cargada de la clase '{images[0]}': {img_array.shape}")

            except Exception as e:
                print(f"Ocurrio un error al cargar la imagen de muestra de la clase '{images[0]}': {e}")
                return False
        else: 
            print (f"La clase '{class_name}' no contiene imagenes validas.")
            return False
            
        total_images += len(images)
        
    print(f"Total de imágenes validas: {total_images}")

    if total_images < len(classes) * 10:
        print("Se necesitan al menos 10 imágenes por clase para entrenar.")

    return True
        

# IMPORTANT: get correct weight

def get_class_weights(dataset_path):

    all_labels = []

    classes = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

    print("Calculando pesos de clase...")

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        num_images = len([f for f in os.listdir(class_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        all_labels.extend([class_idx] * num_images)
        print(f"Clase '{class_name}' (índice {class_idx}): {num_images} imágenes")

    # calc weights 

    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)

    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Pesos calculados: {class_weight_dict}")
    return class_weight_dict

# DEBUGGER

def debug_data_generators(dataset_path): 
    print("Debuggeando Generadores...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
    )
    
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
        shuffle=False
    )
    
    print(f"Clases encontradas: {train_generator.class_indices}")
    print(f"Número de imágenes de entrenamiento: {train_generator.samples}")
    print(f"Número de imágenes de validación: {validation_generator.samples}")
    
    # Verificar un batch
    try:
        x_batch, y_batch = next(train_generator)
        print(f"Forma del batch: {x_batch.shape}")
        print(f"Rango de pixeles: [{x_batch.min():.3f}, {x_batch.max():.3f}]")
        print(f"Ejemplo de etiqueta one-hot: {y_batch[0]}")
        print(f"Clase predicha: {np.argmax(y_batch[0])}")
    except Exception as e:
        print(f"Error al cargar batch: {e}")
    
    class_weights = get_class_weights(dataset_path)
    
    return train_generator, validation_generator, class_weights

# data augmentation and preprocessing

def load_and_preprocess_data(dataset_path):

    # verify dataset structure

    if not verify_dataset_integrity(dataset_path):
        return None, None, None
    
    # verify if theres enough data for the training

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=(0.7, 1.3),
        channel_shift_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # load training data

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode='rgb',
        shuffle=True,
        seed=42
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb',
        shuffle=False
    )

    class_weights= get_class_weights(dataset_path)

    return train_generator, validation_generator, class_weights

# create model

def create_base_model(num_classes):
    # pre trained model 

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
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    print("Modelo creado con MobileNetV2 como base.")
    return model

# train model

def train_model(dataset_path):

    # DEBUGGER

    train_gen, val_gen, class_weights = debug_data_generators(dataset_path)

    if train_gen is None:

        print("No se encontraron datos de entrenamiento. Verifique la estructura de carpetas.")
        return None, None, None, None, None
    
    num_classes= len(train_gen.class_indices)

    print(f"Entrenando modelo con {num_classes} clases...")

    model= create_base_model(num_classes)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # callbacks

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=12, 
            restore_best_weights=True,
            min_delta=0.005,
            mode='max',
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            min_delta=0.01,
        )
    ]
    
    print("Iniciando entrenamiento...")

    # first training

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # fine tuning only if base training was good enough

    if history.history['val_accuracy'][-1] > 0.6:  
        print("Iniciando fine-tuning...")
        base_model = model.layers[0]
        base_model.trainable = True
        
        # fine-tuning only for the last 20 layers

        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        history_fine = model.fit(
            train_gen,
            epochs=8,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    else:
        print("Accuracy muy baja, omitiendo fine-tuning.")
        history_fine = None
    
    return model, history, history_fine, train_gen, val_gen

# evaluate model

def evaluate_model(model, val_gen): 
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

    plt.figure(figsize=(14,12))
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
            model, history, history_fine, train_gen, val_gen = train_model(dataset_path)

            if model is None:
                print("El entrenamiento del modelo falló.")
                exit(1)

            # evaluate the model (only if the validation set exists)
            if val_gen and val_gen.samples > 0:
                y_true, y_pred, class_names = evaluate_model(model, val_gen)
            else:
                print("No se evaluó el modelo debido a la falta de datos de validación.")

            # save the model
            model.save(MODEL_SAVE_PATH)
            print(f"Modelo guardado como: {MODEL_SAVE_PATH}")

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