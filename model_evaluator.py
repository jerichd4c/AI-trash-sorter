from sklearn.metrics import classification_report, confusion_matrix
from data_processor import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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