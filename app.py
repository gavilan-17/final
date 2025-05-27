import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(page_title='Clasificador de Atletas', layout='wide', initial_sidebar_state='expanded')

# Función para cargar los datos (corregida para evitar NaN)
def carga_datos():
    try:
        df = pd.read_csv('atletas.csv')

        # Validar columnas necesarias
        columnas_necesarias = ['Edad', 'Peso', 'Volumen_O2_max', 'Atleta']
        if not all(col in df.columns for col in columnas_necesarias):
            st.error(f"El archivo 'atletas.csv' debe contener las columnas: {', '.join(columnas_necesarias)}")
            st.stop()

        # Eliminar espacios y asegurar tipo string
        df['Atleta'] = df['Atleta'].astype(str).str.strip()
        df['Atleta'] = df['Atleta'].map({'fondista': 1, 'velocista': 0})

        # Eliminar filas con NaN
        df = df.dropna(subset=['Atleta', 'Edad', 'Peso', 'Volumen_O2_max'])

        # Convertir a entero
        df['Atleta'] = df['Atleta'].astype(int)

        if df.empty:
            st.error("El archivo 'atletas.csv' no contiene datos válidos después de eliminar filas con NaN.")
            st.stop()

        return df

    except FileNotFoundError:
        st.error("No se encontró el archivo 'atletas.csv'. Asegúrate de subirlo.")
        st.stop()

# Sidebar
def add_sidebar(df):
    st.sidebar.header('Modifica los datos')
    st.sidebar.title('Parámetros del modelo')
    max_depth = st.sidebar.slider('Profundidad máxima del árbol', 2, 4, 3)
    criterion = st.sidebar.selectbox('Criterio de división', ['gini', 'entropy'])

    # Validar que haya datos válidos para los sliders
    if df['Edad'].dropna().empty or df['Peso'].dropna().empty or df['Volumen_O2_max'].dropna().empty:
        st.error("El archivo 'atletas.csv' no contiene información válida en 'Edad', 'Peso' o 'Volumen_O2_max'.")
        st.stop()

    edad = int(df['Edad'].dropna().min())
    peso = int(df['Peso'].dropna().min())
    volumen_o2 = float(df['Volumen_O2_max'].dropna().min())

    edad = st.sidebar.slider('Edad', edad, int(df['Edad'].max()), int(df['Edad'].mean()))
    peso = st.sidebar.slider('Peso', int(df['Peso'].min()), int(df['Peso'].max()), int(df['Peso'].mean()))
    volumen_o2 = st.sidebar.slider('Volumen_O2_max', float(df['Volumen_O2_max'].min()), float(df['Volumen_O2_max'].max()), float(df['Volumen_O2_max'].mean()))

    return max_depth, criterion, edad, peso, volumen_o2

# Función para entrenar el modelo
def entrena_modelo(df, max_depth, criterion):
    X = df[['Edad', 'Peso', 'Volumen_O2_max']]
    y = df['Atleta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Página de clasificación
def pagina_clasificador():
    df = carga_datos()
    max_depth, criterion, edad, peso, volumen_o2 = add_sidebar(df)
    model, X_test, y_test = entrena_modelo(df, max_depth, criterion)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.title('Clasificación de Atletas')
    pagina = st.sidebar.radio("Selecciona una página:", ['Métricas y Predicción', 'Gráficos'])

    if pagina == 'Métricas y Predicción':
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader('Métricas del Modelo')
            st.write(f'Precisión del modelo: {accuracy:.2f}')
            st.text(classification_report(y_test, y_pred))

        with col2:
            st.subheader('Predicción del Modelo')
            datos_usuario = pd.DataFrame([[edad, peso, volumen_o2]], columns=['Edad', 'Peso', 'Volumen_O2_max'])
            prediccion = model.predict(datos_usuario)[0]
            clase_predicha = 'Fondista' if prediccion == 1 else 'Velocista'
            st.write(f'Según el modelo, el atleta es un: **{clase_predicha}**')

    elif pagina == 'Gráficos':
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Matriz de Confusión')
            fig, ax = plt.subplots(figsize=(5, 3))
            conf_matrix = confusion_matrix(y_test, y_pred)
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', ax=ax)
            st.pyplot(fig)

            st.subheader('Curva ROC-AUC')
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc='lower right')
            st.pyplot(fig)

        with col2:
            st.subheader('Distribución de Predicciones')
            fig, ax = plt.subplots(figsize=(5, 3))
            pd.Series(y_pred).value_counts().plot(kind='bar', color=['blue', 'green'], ax=ax)
            ax.set_xticklabels(['Velocista', 'Fondista'], rotation=0)
            ax.set_xlabel('Clase Predicha')
            ax.set_ylabel('Frecuencia')
            st.pyplot(fig)

            st.subheader('Árbol de Decisión')
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_tree(model, filled=True, feature_names=['Edad', 'Peso', 'Volumen_O2_max'], class_names=['Velocista', 'Fondista'], ax=ax)
            st.pyplot(fig)

# Menú principal
def main():
    st.sidebar.title('Menú Principal')
    pagina = st.sidebar.radio("Navegar a:", ['Clasificador'])
    if pagina == 'Clasificador':
        pagina_clasificador()

if __name__ == '__main__':
    main()
