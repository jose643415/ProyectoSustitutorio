# 🧠 Proyecto Sustitutorio - Modelos I  
**Universidad de Antioquia**

> _“Llevando un modelo predictivo desde la teoría hasta su preparación para producción.”_

---

## 🎯 Objetivo

Este proyecto tiene como finalidad completar el proceso formativo en el curso **Modelos I**, mediante la selección y preparación de un modelo predictivo para su futura integración en un sistema productivo.

---

## 📚 Descripción del Proyecto

Durante el semestre, se desarrollarán distintos entregables enfocados en las etapas prácticas del ciclo de vida de un modelo predictivo. Estas fases incluyen:

### 🚦 FASE 1: Modelo Predictivo
- Selección de un modelo ya construido
- Análisis y documentación básica
- Preparación del entorno de trabajo

📌 Modelo elegido: [AI vs Human - ResNet50V2 (TensorFlow2)](https://www.kaggle.com/models/utkarshsaxenadn/ai-vs-human/TensorFlow2/default/1)  
🔍 Clasifica imágenes como generadas por IA o por humanos.

### 📦 FASE 2: Despliegue en Docker
- Crear un contenedor Docker para ejecutar el modelo
- Configurar dependencias y entorno
- Probar funcionalidad localmente

### 🌐 FASE 3: API REST
- Crear un script en Python para exponer el modelo como API REST
- Permitir el consumo del modelo desde clientes externos
- Validar el servicio con pruebas

---

## 🗂️ Entregables del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| ✅ FASE 1 | Selección y justificación del modelo | Completado |
| ✅ FASE 2 | Despliegue del modelo en contenedor Docker | Completado |
| 🔧 FASE 3 | Creación de API REST para inferencia | En desarrollo |

---

## 🛠️ Instrucciones para ejecutar el código en su primera fase

### 🐍 Requisitos previos

- **Versión de Python recomendada**: `3.11`
- Se recomienda el uso de un entorno virtual (`venv`, `conda`, etc.).
- Instalar las dependencias necesarias: kagglehub, numpy, pandas, matplotlib, tensorflow, keras, cv2 (OpenCV), plotly, Ipython.

### ⚙️ Configuración de la API de Kaggle

El modelo se descarga automáticamente desde Kaggle, por lo que es necesario configurar tu token de autenticación de Kaggle.

### 🖥️ Opción 1: Ejecución local

1. Inicia sesión en [Kaggle](https://www.kaggle.com/).
2. Ve a tu perfil → **Account**.
3. Haz clic en **"Create New API Token"** para descargar el archivo `kaggle.json`.
4. Coloca el archivo en la ruta correspondiente según tu sistema operativo:

   - **Linux/MacOS:**  
     `~/.kaggle/kaggle.json`

   - **Windows:**  
     `C:\Users\<TU_USUARIO>\.kaggle\kaggle.json`

> 🔐 Asegúrate de que la carpeta `.kaggle` tenga los permisos correctos (por ejemplo, 600 en Linux/Mac).

### ☁️ Opción 2: Ejecución en Google Colab

En Colab, puedes subir el archivo `kaggle.json` directamente desde tu máquina para autenticarte temporalmente durante la sesión. Usa el siguiente bloque de código antes de descargar el modelo:

## 📜 Script para configurar Kaggle 

```python
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
```

📌 Recuerda ejecutar este bloque antes de llamar a cualquier función que descargue datos o modelos desde Kaggle.

✅ Ejecutar el script principal para realizar las predicciones: ```python predict_ai_vs_human``` . 

## 🛠️ Instrucciones para ejecutar el código en su segunda fase

### 🐍 Requisitos previos 

Antes de ejecutar este proyecto, asegúrate de cumplir con los siguientes requisitos:
- Tener instalado Docker, puedes hacerlo desde: https://docs.docker.com/get-started/get-docker/
- Tener una cuenta en Kaggle
- Descargar el archivo kaggle.json desde tu cuenta de Kaggle:
  - Ir a tu perfil de Kaggle → Account → Create New API Token
  - Se descargará un archivo llamado kaggle.json

### 📁 Preparación

Al igual que en los pasos para la ejecución de la primera parte de proyecto es necesario que tu archivo .kaggle este debidamente configurado en tu entorno local para poder descargar los datos y el modelo desde Kaggle de manera correcta.

### 🐳 Ejecución del proyecto mediante Docker:

Para ejecutar el proyecto, solo sigue estos pasos:

1. Asegúrate de tener Docker instalado y tu archivo `kaggle.json` ubicado y accesible en tu máquina.

2. Abre una terminal (PowerShell en Windows o terminal en Linux/macOS) y ejecuta:

   - Descargar la imagen desde Docker Hub:
   ```bash
   docker pull josee/ai-human-detector:latest
   ```
   - Ejecutar el contenedor, montando tu carpeta `.kaggle` y exponiendo el puerto 8080:
   ```bash
   docker run --rm -v <Tu_ruta_del_.kaggle>:/root/.kaggle -p 8080:8080 jose643415/ia-human-detector:latest -c "python predict_ai_vs_human.py && python -m http.server 8080"
   ```
3. Luego, abre en tu navegador:
```bash
localhost:8080/predicciones.html
```
4. ¡Listo! Verás las imágenes y predicciones generadas por el modelo.

**Nota**: Tener en cuenta que debido al volumen de datos que contiene el dataset la descarga puede variar en función de tu conexión a internet 

---

## 👥 Créditos y Reconocimientos

- 📦 **Modelo**: [AI vs Human - ResNet50V2 (TensorFlow2)](https://www.kaggle.com/models/utkarshsaxenadn/ai-vs-human/TensorFlow2/default/1)  
  📌 Autor del modelo: [**DeepNets** (usuario en Kaggle)](https://www.kaggle.com/utkarshsaxenadn)

- 🧾 **Dataset**: [AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)  
  👥 Autores del dataset:  
  **Alessandra Sala**, **Manuela Jeyaraj**, **Toma Ijatomi** y **Margarita Pitsiani**

> Agradecimientos a los autores y a la comunidad de **Kaggle** por su contribución al acceso abierto de modelos y datos que permiten el desarrollo de proyectos académicos.

---

## 🧪 Estado Actual

- ⚙️ Se ha cargado y probado el modelo en local con TensorFlow
- ⚙️ Se ha llevado y probado el modelo en un entorno de Docker
- ⌛ Próximo paso: crear una API para consumir el modelo

---

## 📬 Contacto

**Profesor:** Raúl Ramos 
**Estudiantes:** José Andrés Echavarría Ríos  
**Curso:** Modelos I - Universidad de Antioquia

---

## 📌 Notas Finales

Este proyecto es de carácter **educativo** y tiene como propósito fortalecer habilidades prácticas en ciencia de datos y despliegue de modelos. No se busca alcanzar métricas de producción, sino comprender el flujo completo de trabajo.

---

