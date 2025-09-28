from data_processor import *
import matplotlib.pyplot as plt
from base_model import *
from basic_config import *
from data_processor import * 
from model_trainer import *
from base_model import *
from model_evaluator import *

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