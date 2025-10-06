
# Trash Sorter usando IA 🤖

Trash sorter basico (Clasificador de residuos) creado en **Python** usando IA, redes neuronales convolucionales y machine learning para entrenarse ella misma y clasificar diferentes tipos de residuos siguiendo el dataset de Kaggle: [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
12 categorias (ir al link para mas referencia)

## Requerimientos previos ⚠️

Antes de usar el proyecto, tendra que instalar las dependencias almacenadas en el archivo: **requeriments.txt**

Para instalar el archivo, escribe esto en el terminal:

```bash
pip install requeriments.txt
```
    
## Crear el modelo 📦

<div align="center">

❗**IMPORTANTE**❗

</div>

El repositorio ya tiene un modelo pre-entrenado: **waste_classifier_model.h5** sin embargo, en caso de que el archivo se pierda o se quiera recompilar con otros valores, se tiene que realizar lo siguiente: 

Escribir en el terminal:

```bash
python run src\full_model.py 
```

(*dependiendo del dataset y los valores predefinidos el compilador puede tardarse*)

## Ejecutar el proyecto 🚀 

Una vez con el modelo en  el directorio base, se tiene ejecutar en el terminal el siguiente comando:

```bash
python run src\waste_app.py 
```

*NT: esta app usa la libreria **Streamlit**, diseñada para crear UIs simples en *Python**.


## Ejemplos de uso ♻️ 

<div align="center">

**Tipo de desecho: clothes/ropa**

![Example classification 1](results/examples/example_1.png)

**Estadisticas usando modelo preentrenado:**

![Example classification 1 stats](results/examples/example_1_stats.png)

**Tipo de desecho: cardboard/carton**

![Example classification 2](results/examples/example_2.png)

**Estadisticas usando modelo preentrenado:**

![Example classification 2 stats](results/examples/example_2_stats.png)

**Tipo de desecho: battery/bateria**

![Example classification 3](results/examples/example_3.png)

**Estadisticas usando modelo preentrenado:**

![Example classification 3 stats](results/examples/example_3_stats.png)

*NT: Hay tipos de desechos con prediccion mas favorables que otros, si se quiere ser mas exacto en clases especificas, es necesario usar otros datasets*.

---

⚠️ **ADVERTENCIA** ⚠️
</div>

El archivo **waste_classifier_model.h5** tiene que estar en la ruta base del proyecto, de lo contrario, saldra un mensaje indicando que no hay un modelo existente.

