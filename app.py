# app.py

from flask import Flask, request, jsonify, render_template, url_for
import joblib
import bleach
import logging
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from utils.preprocessing import clean_tokenize_lemmatize  # Asegúrate de que este módulo exista y sea correcto

# Inicializar la aplicación Flask
app = Flask(__name__)

# Configuración de Flask-Limiter para limitar solicitudes a 5 por minuto por IP
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=[]  # No establecemos límites globales aquí
)

# Configuración de logging para registrar errores y eventos en 'app.log'
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,  # Cambiado a DEBUG para obtener más detalles durante la depuración
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)

# Log de inicio de la aplicación
logging.info("Iniciando la aplicación Flask")

# Rutas a los archivos de modelos
MODEL_DIR = 'models'
MODEL_FILE = 'random_forest_spam_model.joblib'
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
LABEL_ENCODER_FILE = 'label_encoder.joblib'

model_path = os.path.join(MODEL_DIR, MODEL_FILE)
vectorizer_path = os.path.join(MODEL_DIR, VECTORIZER_FILE)
label_encoder_path = os.path.join(MODEL_DIR, LABEL_ENCODER_FILE)

# Verificar que los archivos de modelos existan
for path, name in [(model_path, "Modelo"), (vectorizer_path, "Vectorizador"), (label_encoder_path, "Label Encoder")]:
    if not os.path.exists(path):
        logging.error(f"{name} no encontrado en la ruta: {path}")
        raise FileNotFoundError(f"{name} no encontrado en la ruta: {path}")

# Cargar los modelos, vectorizador y codificador de etiquetas
try:
    logging.info("Cargando el modelo Random Forest")
    rf_model = joblib.load(model_path)
    logging.info("Modelo Random Forest cargado exitosamente")

    logging.info("Cargando el vectorizador TF-IDF")
    vectorizer = joblib.load(vectorizer_path)
    logging.info("Vectorizador TF-IDF cargado exitosamente")

    logging.info("Cargando el Label Encoder")
    label_encoder = joblib.load(label_encoder_path)
    logging.info("Label Encoder cargado exitosamente")
except Exception as e:
    logging.exception("Error al cargar los modelos")
    raise e

# Asegurarse de que los recursos de NLTK estén descargados
import nltk

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    logging.info("Recursos de NLTK encontrados")
except LookupError:
    logging.info("Descargando recursos de NLTK")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        logging.info("Recursos de NLTK descargados exitosamente")
    except Exception as e:
        logging.exception("Error al descargar los recursos de NLTK")
        raise e

# Ruta principal para la interfaz web
@app.route('/')
def home():
    logging.debug("Cargando la página principal")
    return render_template('index.html')

# Ruta para manejar la predicción con limitación de solicitudes
@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")  # Limitar a 5 solicitudes por minuto por IP
def predict():
    logging.debug("Solicitud de predicción recibida")
    try:
        # Obtener el texto del formulario
        texto = request.form.get('texto', '')

        logging.debug(f"Texto recibido para análisis: {texto}")

        # Validación: Verificar si el texto está vacío
        if not isinstance(texto, str) or texto.strip() == "":
            logging.warning("Entrada vacía recibida")
            return jsonify({'error': 'La entrada debe ser una cadena de texto no vacía.'}), 400

        # Sanitización: Limpiar el texto para eliminar posibles códigos maliciosos
        texto_sanitizado = bleach.clean(texto)
        logging.debug(f"Texto sanitizado: {texto_sanitizado}")

        # Preprocesar el texto
        texto_limpio = clean_tokenize_lemmatize(texto_sanitizado)
        logging.debug(f"Texto preprocesado: {texto_limpio}")

        # Verificar que el texto preprocesado no esté vacío
        if not texto_limpio:
            logging.warning("Texto preprocesado está vacío después del preprocesamiento")
            return jsonify({'error': 'El texto proporcionado no es válido después del preprocesamiento.'}), 400

        # Vectorizar el texto
        vectorizado = vectorizer.transform([texto_limpio])
        logging.debug(f"Texto vectorizado: {vectorizado}")

        # Predicción de probabilidad
        probabilidad = rf_model.predict_proba(vectorizado)[0][1]  # Probabilidad de ser SPAM
        porcentaje = probabilidad * 100
        logging.debug(f"Probabilidad calculada de SPAM: {porcentaje:.2f}%")

        # Determinar el mensaje basado en la probabilidad
        if porcentaje < 20:
            mensaje = f"Probabilidad de SPAM: {porcentaje:.2f}%. Es muy probable que NO sea SPAM."
            is_spam = False
        elif 20 <= porcentaje < 50:
            mensaje = f"Probabilidad de SPAM: {porcentaje:.2f}%. Es probable que NO sea SPAM."
            is_spam = False
        elif 50 <= porcentaje < 80:
            mensaje = f"Probabilidad de SPAM: {porcentaje:.2f}%. Es probable que SÍ sea SPAM."
            is_spam = True
        else:
            mensaje = f"Probabilidad de SPAM: {porcentaje:.2f}%. Es muy probable que SÍ sea SPAM."
            is_spam = True

        logging.debug(f"Mensaje generado: {mensaje}, is_spam: {is_spam}")

        # Retornar el resultado como JSON
        return jsonify({'message': mensaje, 'is_spam': is_spam})

    except Exception as e:
        # Registrar el error en el log con traceback completo
        logging.exception("Error en la ruta /predict")
        # Retornar un error genérico al cliente
        return jsonify({'error': 'Ocurrió un error interno en el servidor.'}), 500

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    try:
        logging.info("Ejecutando la aplicación Flask")
        app.run(debug=True)  # Temporariamente establecido en True para depuración
    except Exception as e:
        logging.exception("Error al iniciar la aplicación Flask")
        raise e
