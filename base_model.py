from keras._tf_keras.keras.applications import EfficientNetB0
from basic_image_config import IMG_SIZE, BATCH_SIZE, DATASET_PATH, NUM_CLASSES

# pre trained model 

def create_base_model(NUM_CLASSES):
    base_model = EfficientNetB0( 
        weights='imagenet',
        include_top=False,
        input_shape=(220, 220, 3)
    )

    base_model.trainable = False
    
    return base_model   