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

# ─── LIBRERÍAS ────────────────────────────────────────────────────────────────
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ─── DESCARGA DE DATOS Y MODELO ─────────────────────────────────────
alessandrasala79_ai_vs_human_generated_dataset_path = kagglehub.dataset_download(
    'alessandrasala79/ai-vs-human-generated-dataset')

utkarshsaxenadn_ai_vs_human_tensorflow2_default_1_path = kagglehub.model_download(
    'utkarshsaxenadn/ai-vs-human/TensorFlow2/default/1')

print("✔️ Datos y modelo descargados correctamente.")

# ─── CONFIGURACIÓN ──────────────────────────────────────────────────
IMG_SIZE = 512  # Según el modelo ResNet50V2
LABELS = ["Human", "AI"]

# ─── CARGAR MODELO ──────────────────────────────────────────────────
# Construir la ruta completa al archivo del modelo usando el directorio descargado.
model_file = os.path.join(utkarshsaxenadn_ai_vs_human_tensorflow2_default_1_path, "ResNet50V2-AIvsHumanGenImages.keras")
model = tf.keras.models.load_model(model_file)
print("✔️ Modelo cargado correctamente.")

# ─── FUNCIÓN DE CARGA Y PREDICCIÓN ──────────────────────────────────
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ─── VISUALIZACIÓN Y TABLA ──────────────────────────────────────────
def show_images_and_predictions(image_paths, model, n_rows=3, n_cols=5):
    predictions_data = []
    n_images = min(len(image_paths), n_rows * n_cols)

    plt.figure(figsize=(15, 10))

    for idx in range(n_images):
        img_path = image_paths[idx]
        img_tensor = load_and_preprocess(img_path)
        prediction = model.predict(img_tensor, verbose=0)[0][0]

        predicted_label = "AI" if prediction > 0.5 else "Human"
        true_label = "AI" if "AI" in os.path.basename(img_path).upper() else "Human"

        predictions_data.append({
            "ID": idx + 1,
            "Predicción": predicted_label,
            "Etiqueta Real": true_label,
            "Probabilidad AI": round(prediction, 2)
        })

        # Mostrar imagen con ID
        plt.subplot(n_rows, n_cols, idx + 1)
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"ID: {idx + 1}", fontsize=10)

    plt.tight_layout()
    plt.show()

    # Mostrar tabla de predicciones
    df = pd.DataFrame(predictions_data)
    print("\n📊 Tabla de predicciones:")
    print(df.to_string(index=False))

# ─── CARGAR IMÁGENES ────────────────────────────────────────────────
test_folder = os.path.join(alessandrasala79_ai_vs_human_generated_dataset_path, "test_data_v2")
image_paths = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.lower().endswith(".jpg")]

# ─── EJECUTAR VISUALIZACIÓN Y PREDICCIÓN ───────────────────────────
show_images_and_predictions(image_paths[:30], model, n_rows=10, n_cols=10)


