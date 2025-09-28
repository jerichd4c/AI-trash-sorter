import keras
from keras._tf_keras.keras.optimizers import Adam
from base_model import *
from basic_config import *
from data_processor import *
from data_processor import *
from base_model import *
from basic_config import *
from base_model import *

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
