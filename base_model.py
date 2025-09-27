import keras
from keras._tf_keras.keras.applications import EfficientNetB0
from keras import layers
from basic_config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, NUM_CLASSES

# pre trained model 

def create_base_model(NUM_CLASSES):
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