# entrenamiento_modelo.py

# -*- coding: utf-8 -*-
"""Entrenamiento del Modelo de Detección de Spam"""

import pandas as pd
import numpy as np
import re
import nltk
import gdown
import joblib  # Para guardar el modelo y el vectorizador
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Instalación necesaria (asegúrate de ejecutar esto en tu entorno)
# !pip install nltk gdown joblib

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Función de limpieza, tokenización y lematización
def clean_tokenize_lemmatize(text):
    # Eliminamos caracteres especiales
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\.+$', '', text)  # Eliminamos puntos al final
    text = re.sub(r'(.)\1{2,}', '', text)  # Eliminamos secuencias repetidas de más de dos caracteres
    text = re.sub(r'\b(free|win|prize|money)\b', '', text)  # Eliminamos ciertas palabras claves
    text = re.sub(r'https?://\S+', '', text)  # Eliminamos enlaces
    text = re.sub(r'[0-9]{2}/[0-9]{2}/[0-9]{4}', '', text)  # Eliminamos fechas
    text = text.lower()  # Convertimos a minúsculas

    # Tokenización
    tokens = word_tokenize(text)

    # Eliminamos stopwords y símbolos no alfabéticos
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Lematización
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(lemmatized_tokens)

# Función para crear el DataFrame Bag of Words
def create_bow_dataframe(df, vectorizer=None, fit=True):
    if fit:
        X = vectorizer.fit_transform(df['text'])
    else:
        X = vectorizer.transform(df['text'])
    bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

def main():
    """Proceso de Entrenamiento"""

    # URLs de descarga de los datasets
    datasets = {
        'sms': '1sTaQb34pYFzNj7MNqs1OymfH9P1Mo7m0',
        'tlg': '174i0fnWHh08lpJXNU1MstGvqiCTZFTTI',
        'enron': '147V7I0LSOs3mc1kI7Ac3suFEnLGnYGXv'
    }

    # Descargar y cargar datasets
    dataframes = []
    for name, file_id in datasets.items():
        download_url = f'https://drive.google.com/uc?id={file_id}'
        output = f'{name}.csv'
        gdown.download(download_url, output, quiet=False)
        df = pd.read_csv(output, encoding='latin-1')
        dataframes.append(df)

    # Renombrar columnas y limpiar datasets
    sms, tlg, enron = dataframes

    sms.rename(columns={'v1': 'Ham/Spam', 'v2': 'text'}, inplace=True)
    sms.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

    tlg.rename(columns={'text_type': 'Ham/Spam'}, inplace=True)

    enron.rename(columns={'Spam/Ham': 'Ham/Spam', 'Message': 'text'}, inplace=True)
    enron.drop(columns=['Date'], inplace=True)
    enron.dropna(inplace=True)

    # Concatenar datasets
    data = pd.concat([sms, tlg, enron], ignore_index=True)

    # Combinar 'Subject' y 'text' si 'Subject' no es nulo
    if 'Subject' in data.columns:
        mask = data['Subject'].notnull()
        data.loc[mask, 'text'] = data.loc[mask, 'Subject'] + ' ' + data.loc[mask, 'text']
        data.drop(columns=['Subject'], inplace=True)

    # Limpiar y preprocesar texto
    data['text'] = data['text'].apply(clean_tokenize_lemmatize)

    # Vectorización usando TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000)
    bow_df = create_bow_dataframe(data, vectorizer=vectorizer, fit=True)

    # Preparar variables predictoras y objetivo
    X = bow_df
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Ham/Spam'])

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluación del modelo (opcional)
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred_rf = rf_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    # Guardar el modelo y el vectorizador
    joblib.dump(rf_model, 'random_forest_spam_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    print("Modelos y vectorizador guardados exitosamente.")

if __name__ == "__main__":
    main()
