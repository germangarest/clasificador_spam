# utils/preprocessing.py

import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Inicializar recursos una sola vez para mejorar el rendimiento
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_tokenize_lemmatize(text):
    """
    Limpia, tokeniza y lematiza el texto proporcionado.

    Parámetros:
        text (str): El texto a procesar.

    Retorna:
        str: El texto procesado listo para la vectorización.
    """
    try:
        # Eliminamos caracteres especiales y números, dejando solo letras y espacios
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        
        # Eliminamos puntos al final de las oraciones
        text = re.sub(r'\.+$', '', text)
        
        # Eliminamos secuencias repetidas de más de dos caracteres (e.g., "soooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Eliminamos ciertas palabras clave que pueden ser indicativas de spam
        text = re.sub(r'\b(free|win|prize|money)\b', '', text, flags=re.IGNORECASE)
        
        # Eliminamos enlaces web
        text = re.sub(r'https?://\S+', '', text)
        
        # Eliminamos fechas en formato dd/mm/yyyy
        text = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', '', text)
        
        # Convertimos el texto a minúsculas
        text = text.lower()

        # Tokenización
        tokens = word_tokenize(text)

        # Eliminamos stopwords y símbolos no alfabéticos
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

        # Lematización
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Unimos los tokens lematizados en una sola cadena de texto
        return ' '.join(lemmatized_tokens)
    except Exception as e:
        logging.exception("Error durante el preprocesamiento del texto")
        # En caso de error, devolver una cadena vacía o manejar de otra manera
        return ''
