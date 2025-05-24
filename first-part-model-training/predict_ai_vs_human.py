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
import http.server
import socketserver

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

def show_images_and_predictions_html(image_paths, model, output_file="predicciones.html", n_images=20):
    predictions_data = []
    n_images = min(len(image_paths), n_images)
    html = ['<html><head><title>predictions</title></head><body>']
    html.append('<h1>Image predictions</h1>')
    html.append('<table border="1" style="border-collapse: collapse;"><tr><th>ID</th><th>Imagen</th>'
                '<th>predictions</th><th>Real Label</th><th>Probability of Ia</th></tr>')

    for idx in range(n_images):
        img_path = image_paths[idx]
        img_tensor = load_and_preprocess(img_path)
        prediction = model.predict(img_tensor, verbose=0)[0][0]

        predicted_label = "AI" if prediction > 0.5 else "Human"
        true_label = "AI" if "AI" in os.path.basename(img_path).upper() else "Human"

        predictions_data.append((idx + 1, img_path, predicted_label, true_label, round(prediction, 2)))

        # Leer imagen como base64
        import base64
        from io import BytesIO
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        buf = BytesIO()
        img.save(buf, format='JPEG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        html.append(f'<tr><td>{idx + 1}</td><td><img src="data:image/jpeg;base64,{img_b64}" width="100"/></td>'
                    f'<td>{predicted_label}</td><td>{true_label}</td><td>{round(prediction, 2)}</td></tr>')

    html.append('</table></body></html>')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

    print(f"✅ Archivo HTML generado: {output_file}")


# ─── CARGAR IMÁGENES ────────────────────────────────────────────────
test_folder = os.path.join(alessandrasala79_ai_vs_human_generated_dataset_path, "test_data_v2")
image_paths = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.lower().endswith(".jpg")]

# ─── EJECUTAR VISUALIZACIÓN Y PREDICCIÓN ───────────────────────────
#show_images_and_predictions(image_paths[:20], model, n_rows=10, n_cols=10)
output_html = "/app/predicciones.html"
show_images_and_predictions_html(image_paths[:20], model)

print("🌐 Servidor web iniciado en http://localhost:8080/predicciones.html")
PORT = 8080
handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("", PORT), handler)
httpd.serve_forever()