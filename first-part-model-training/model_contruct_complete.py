# ─── KAGGLE CONFIGURACIÓN ─────────────────────────────────────────────────────
import os
import shutil
# Ruta donde se debe ubicar kaggle.json
kaggle_dir = os.path.expanduser('~/.kaggle')
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

# Si estás en Google Colab, sube el archivo
try:
    from google.colab import files
    uploaded = files.upload()  # Subir kaggle.json
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.move('kaggle.json', kaggle_json_path)
    os.chmod(kaggle_json_path, 0o600)
except ImportError:
    # Para entornos locales: asegúrate de que kaggle.json ya esté en ~/.kaggle/
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError("Por favor coloca tu kaggle.json en ~/.kaggle/")


# ─── KAGGLEHUB: Descargar Dataset y Modelo ─────────────────────────────────────
import kagglehub

# Descargar el dataset
alessandrasala79_ai_vs_human_generated_dataset_path = kagglehub.dataset_download(
    'alessandrasala79/ai-vs-human-generated-dataset'
)

# Descargar el modelo
utkarshsaxenadn_ai_vs_human_tensorflow2_default_1_path = kagglehub.model_download(
    'utkarshsaxenadn/ai-vs-human/TensorFlow2/default/1'
)

print('✔️ Datos y modelo descargados correctamente.')

# ─── IMPORTS GENERALES ────────────────────────────────────────────────────────
import cv2
import keras
import numpy as np
import tensorflow as tf

# Data Loading
import pandas as pd
import tensorflow as tf
import tensorflow.data as tfd
import tensorflow.image as tfi

# Data Visualization
import plotly.express as px
import matplotlib.pyplot as plt

# Pre Trained Models
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import Xception, InceptionV3
from tensorflow.keras.applications import ResNet152, ResNet152V2
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB5

# Outputs
from IPython.display import clear_output as cls

# Plotly Configuration
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import plotly.io as pio
pio.renderers.default = 'browser'

#Data visualizacion
def show_images(data, n_rows=3, n_cols=5, figsize=(15, 10)):
    # Get the images and labels
    images, labels = next(iter(data))

    # Create subplots
    plt.figure(figsize=figsize)
    n_image = 0  # Initialize the image index

    # Loop through the grid and plot images
    for i in range(n_rows):
        for j in range(n_cols):
            if n_image < len(images):  # Ensure we don't index beyond the number of images
                plt.subplot(n_rows, n_cols, n_image + 1)
                plt.imshow(images[n_image])  # Display the image
                plt.axis('off')
                plt.title("AI" if labels[n_image].numpy()==1.0 else "Human")  # Convert label to string for display
                n_image += 1

    plt.show()

train_csv_path = os.path.join(alessandrasala79_ai_vs_human_generated_dataset_path, "train.csv")
test_csv_path = os.path.join(alessandrasala79_ai_vs_human_generated_dataset_path, "test.csv")
main_dir = alessandrasala79_ai_vs_human_generated_dataset_path
history_df_path = os.path.join(utkarshsaxenadn_ai_vs_human_tensorflow2_default_1_path, "history_df.csv")

# ─── SEMILLA PARA REPRODUCIBILIDAD ────────────────────────────────────────────
seed = 42
tf.random.set_seed(seed)

# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────
train_csv = pd.read_csv(train_csv_path, index_col=0)
test_csv = pd.read_csv(test_csv_path, index_col=0)

# Vista rápida de los datos
print("Train CSV:")
# Quick Look
print(train_csv.head())
# Testing CSV Quick look
test_csv.head()

# Size of training and testing data
print(f"Number of training samples: {train_csv.shape[0]}")
print(f"Number of testing samples : {test_csv.shape[0]}")

# Compute Class Distribution
class_dis = train_csv.label.value_counts().reset_index()

# Visualization
pie_fig = px.pie(
    class_dis,
    names='label',
    values='count',
    title='Class Distribution',
    hole=0.2,
)
#pie_fig.show()

bar_fig = px.bar(
    class_dis,
    x='label',
    y='count',
    title='Class Distribution',
    text='count'
)
bar_fig.update_layout(
    xaxis_title='Label',
    yaxis_title='Frequency Count'
)
#bar_fig.show()

for i in range(10):
    shape = plt.imread(f'{main_dir}/{train_csv.file_name[i]}').shape
    print(f"Shape of the Image: {shape}")

# Collect Training image paths
image_paths, image_labels = train_csv.file_name, train_csv.label

# Define split sizes (80% train, 10% val, 10% test)
total_size = len(image_paths)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = int(0.1 * total_size)

print(f"Training Data   : {train_size}")
print(f"Testing Data    : {test_size}")
print(f"Validation Data : {val_size}")

train_paths, train_labels = image_paths[:train_size], image_labels[:train_size]
val_paths, val_labels = image_paths[train_size:train_size+val_size], image_labels[train_size:train_size+val_size]
test_paths, test_labels = image_paths[train_size+val_size:], image_labels[train_size+val_size:]

BATCH_SIZE = 32

# Updated generator function that works for train, val, and test sets
def image_generator(file_paths, labels):
    for file_path, label in zip(file_paths, labels):
        file_path = main_dir + '/' + file_path
        image = cv2.imread(file_path)  # Read using OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (512, 512))  # Resize
        image = image.astype(np.float32) / 255.0  # Normalize
        yield image, label  # Yield image and label

# Function to create datasets
def create_dataset(file_paths, labels, batch_size=BATCH_SIZE):
    return tf.data.Dataset.from_generator(
        lambda: image_generator(file_paths, labels),
        output_signature=(
            tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Creating train, validation, and test datasets
train_ds = create_dataset(train_paths, train_labels)
val_ds = create_dataset(val_paths, val_labels)
test_ds = create_dataset(test_paths, test_labels)

# Visualize Images
#show_images(val_ds)

pre_trained_models = {
    'ResNet50':ResNet50(include_top=False, input_shape=(512, 512, 3)),
    'ResNet50V2':ResNet50V2(include_top=False, input_shape=(512, 512, 3)),
    'Xception':Xception(include_top=False, input_shape=(512, 512, 3)),
    'InceptionV3':InceptionV3(include_top=False, input_shape=(512, 512, 3)),
    'ResNet152':ResNet152(include_top=False, input_shape=(512, 512, 3)),
    'ResNet152V2':ResNet152V2(include_top=False, input_shape=(512, 512, 3)),
    'EfficientNetB3':EfficientNetB3(include_top=False, input_shape=(512, 512, 3)),
    'EfficientNetB5':EfficientNetB5(include_top=False, input_shape=(512, 512, 3))
}

if os.path.exists(history_df_path):
    history_df = pd.read_csv(history_df_path, index_col=0)
else:
    histories = {}
    for model_name in pre_trained_models:
        model = pre_trained_models[model_name]
        print(f"Utilizing : {model_name}")

        # Freeze Model Weights
        model.trainable = False

        # Create another model
        base_model = keras.Sequential([
            model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        base_model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Fit the Model on training data
        history = base_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
        )
        histories[model_name] = history

        cls()
        del base_model
        del history

    # Convert the History callback into Histories
    for name in histories:
        histories[name] = histories[name].history

    # Conver the History in DataFrame
    history_df = pd.DataFrame(histories).T
    history_df = history_df.reset_index()
    history_df = history_df.rename(columns={'index': 'Model'})

    # Save to CSV File
    history_df.to_csv('history_df.csv')

#print(history_df)

resnet50v2_path = '/kaggle/input/ai-vs-human/tensorflow2/default/1/ResNet50V2-AIvsHumanGenImages.keras'

if os.path.exists(resnet50v2_path):
    # Load the Model
    resnet50_model = keras.models.load_model(resnet50v2_path, compile=True)
    # Freeze Model Weights
    resnet50_model.trainable = False
else:
    # Load the Pre Trained Model
    resnet50_base_model = ResNet50V2(
        input_shape=(512, 512, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model weights
    resnet50_base_model.trainable = False

    # Creating Model
    resnet50_model = keras.Sequential([
        resnet50_base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation='sigmoid')
    ], name='ResNet50V2-AIvsHumanGenImages')

    # Compile the model
    resnet50_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )

# Model Summary
resnet50_model.summary()

resnet50v2_history_path = '/kaggle/input/ai-vs-human/tensorflow2/default/1/ResNet50V2-AIvsHumanGenImages-Logs.csv'

if os.path.exists(resnet50v2_history_path):
    # Load the logs
    resnet50v2_history = pd.read_csv(resnet50v2_history_path)
else:
    # Model Training
    resnet50_model_history = resnet50_model.fit(
        train_ds,
        epochs=100,
        steps_per_epoch=train_size//BATCH_SIZE,
        validation_data=val_ds,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
            keras.callbacks.ModelCheckpoint('ResNet50V2-AIvsHumanGenImages.keras', save_best_only=True),
            keras.callbacks.CSVLogger('ResNet50V2-AIvsHumanGenImages-Logs.csv', append=True),
            keras.callbacks.TerminateOnNaN()
        ]
    )

# Quick Look
resnet50v2_history.head()

resnet152_model_path = '/kaggle/input/ai-vs-human/tensorflow2/default/1/ResNet152V2-AIvsHumanGenImages.keras'

if os.path.exists(resnet152_model_path):
    resnet152_model = keras.models.load_model(resnet152_model_path, compile=True)
    resnet152_model.trainable = False
else:
    # Load the Pre Trained Model
    resnet152_base_model = ResNet152V2(
        input_shape=(512, 512, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model weights
    resnet152_base_model.trainable = False

    # Creating Model
    resnet152_model = keras.Sequential([
        resnet152_base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation='sigmoid')
    ], name='ResNet152V2-AIvsHumanGenImages')

    # Compile the model
    resnet152_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )

# Model Summary
resnet152_model.summary()


resnet152v2_history_path = '/kaggle/input/ai-vs-human/tensorflow2/default/1/ResNet152V2-AIvsHumanGenImages-Logs.csv'

if os.path.exists(resnet50v2_history_path):
    # Load the logs
    resnet152v2_history = pd.read_csv(resnet152v2_history_path)
else:
    # Model Training
    resnet152_model_history = resnet152_model.fit(
        train_ds,
        epochs=100,
        steps_per_epoch=train_size//BATCH_SIZE,
        validation_data=val_ds,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
            keras.callbacks.ModelCheckpoint('ResNet152V2-AIvsHumanGenImages.keras', save_best_only=True),
            keras.callbacks.CSVLogger('ResNet152V2-AIvsHumanGenImages-Logs.csv', append=True),
            keras.callbacks.TerminateOnNaN()
        ]
    )

resnet152v2_history.head()

#resnet50_model.evaluate(test_ds)

#resnet152_model.evaluate(test_ds)