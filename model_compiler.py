import keras
from keras._tf_keras.keras.optimizers import Adam
from base_model import model
from basic_config import EPOCHS
from data_processor import train_generator as train_gen, validation_generator as val_gen
from base_model import model


def compile_model(model):

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():

    callbacks = [ 
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),   
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
    ]
    
    # first training

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

    history_fine = model.fit(
        train_gen,
        epochs=EPOCHS + 10,
        initial_epoch=history.epoch[-1],
        validation_data=val_gen,
        callbacks=callbacks
    )

    return model, history, history_fine, train_gen, val_gen
