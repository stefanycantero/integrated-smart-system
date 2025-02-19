# Herramienta Web: Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en Comercio Electrónico

Este repositorio contiene una aplicación web desarrollada con Streamlit en Python que permite a los usuarios interactuar con modelos de Machine Learning para la predicción de demanda, clasificación de productos y recomendación de productos. Los modelos fueron entrenados utilizando los siguientes conjuntos de datos tomados de Kaggle:

* Series de tiempo: https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data
* Clasificación de imágenes: https://www.kaggle.com/datasets/sunnykusawa/ecommerce-products-image-dataset/data
* Sistemas de recomendación: https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/data
  
Equipo 01 conformado por:
* Sebastian Aguinaga Velasquez
* Stefany Cantero Cárdenas
* María Del Pilar Mira Londoño 
* Verónica Pérez Zea

## Funcionalidades

### Predicción de Demanda
* Permite visualizar gráficos de predicciones de demanda a 30 días utilizando el dataset *Walmart Sales Forecast*.
* Los usuarios pueden subir un conjunto de datos propio y previsualizar los 5 primeros registros.
* Se generan predicciones mostradas en una tabla y una gráfica interactiva.
* El modelo se desarrolló con *PyTorch* utilizando una arquitectura *LSTM*.

### Clasificación de Productos
* Permite a los usuarios subir imágenes de productos y ver la categoría asignada.
* Se clasifica en una de las siguientes categorías: *jeans, sofá, TV, t-shirt* según el dataset *ecommerce-product-image-dataset* o como *desconocido* en caso de no pertenecer a ninguna.
* El modelo se desarrolló con *TensorFlow* utilizando la arquitectura de *CNN*.

### Sistema de Recomendación
* Ofrece recomendaciones personalizadas para diferentes usuarios.
* Se selecciona un producto de una lista (el último con el que el usuario interactuó) basada en el dataset *amazon-products-dataset*.
* Utilizando una matriz de similitud, se obtienen las 5 recomendaciones más similares, mostrando el nombre, categoría, subcategoría, calificación y descripción del producto.

## Tecnologías Utilizadas

* **Framework de desarrollo:** Streamlit (Python)
* **Modelos de Machine Learning:**
  * *PyTorch* para la predicción de demanda (*LSTM*).
  * *TensorFlow* para la clasificación de productos (*CNN*).
  * Algoritmo de similitud para el sistema de recomendación.
* **Librerías de Preprocesamiento y Entrenamiento:**
  * NumPy
  * Pandas
  * Matplotlib
  * Seaborn
  * Scikit-learn
  * Tensorflow
  * Pytorch
  * Pickle

## Estructura del Proyecto

```
├── app.py             # Archivo principal de la aplicación Streamlit
├── demand_prediction.py             # Módulo de predicción de demanda
├── product_classification.py             # Módulo de clasificación de productos
├── recommendation_system.py             # Módulo del sistema de recomendación
├── models/            # Carpeta que contiene los modelos entrenados
|     demand_prediction/
│     ├── modelo_demanda.pth      # Modelo LSTM entrenado en PyTorch
│     ├── demand_example.png      # Imagen de ejemplo de demanda
│     ├── modelodos.ipynb   # Código fuente entrenamiento
│     ├── EDA&modelouno.ipynb      # Código de descripción de datos
│     ├── sales_data (1).csv      # Conjunto de datos
|     product_classification/
│     ├── modelo_cnn.pkl  # Modelo de clasificación en TensorFlow
│     ├── RN_Image_Clasification.ipynb   # Código fuente entrenamiento
|     recommendation_system/
│     ├── SE_usuarioProducto  # Código fuente
│     ├── amazon_products.csv      # Conjunto de datos
├── videos/            # Carpeta que contiene los modelos entrenados
│     ├── Demanda.mp4     # Video explicativo
│     ├── Productos.mp4     # Video explicativo
│     ├── Recomendación.mp4     # Video explicativo
│     ├── General.mp4     # Video explicativo
├── requirements.txt   # Lista de dependencias del proyecto  
└── README.md          # Este archivo  
```
