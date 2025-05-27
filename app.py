import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, accuracy_score
)
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(page_title='Clasificador de Atletas', layout='wide', initial_sidebar_state='expanded')

# --- Funciones auxiliares ---
def carga_datos():
    df = pd.read_csv('atletas.csv')
    df['Atleta'] = df['Atleta'].astype(str).str.strip()
    df['Atleta'] = df['Atleta'].map({'fondista': 1, 'velocista': 0})
    df = df.dropna(subset=['Atleta', 'Edad', 'Peso', 'Volumen_O2_max'])
    df['Atleta'] = df['Atleta'].astype(int)
    return df

def add_sidebar(df):
    st.sidebar.header('Modifica los datos')
    st.sidebar.title('Parámetros del modelo')
    max_depth = st.sidebar.slider('Profundidad máxima del árbol', 2, 4, 3)
    criterion = st.sidebar.selectbox('Criterio de división', ['gini', 'entropy'])

    edad = int(df['Edad'].dropna().min())
    peso = int(df['Peso'].dropna().min())
    volumen_o2 = float(df['Volumen_O2_max'].dropna().min())

    edad = st.sidebar.slider('Edad', edad, int(df['Edad'].max()), int(df['Edad'].mean()))
    peso = st.sidebar.slider('Peso', int(df['Peso'].min()), int(df['Peso'].max()), int(df['Peso'].mean()))
    volumen_o2 = st.sidebar.slider('Volumen_O2_max', float(df['Volumen_O2_max'].min()), float(df['Volumen_O2_max'].max()), float(df['Volumen_O2_max'].mean()))

    return max_depth, criterion, edad, peso, volumen_o2

def entrena_modelo(df, max_depth, criterion):
    X = df[['Edad', 'Peso', 'Volumen_O2_max']]
    y = df['Atleta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# --- Página de Árbol y Gini manual ---
def pagina_arbol_manual():
    st.title('Análisis Manual del Árbol de Decisión y Gini')
    Glucose = [150, 140, 140, 120, 80, 70, 90, 70]
    Age = [50, 45, 60, 70, 75, 40, 80, 80]
    Outcome = [1, 0, 1, 1, 0, 0, 0, 0]
    df = pd.DataFrame({'Glucose': Glucose, 'Age': Age, 'Outcome': Outcome})
    X = df[['Glucose', 'Age']].values
    y = df['Outcome'].values

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Glucose', y='Age', hue='Outcome', palette='viridis', ax=ax)
    ax.axvline(x=120, color='red', linestyle='dashed')
    ax.axhline(y=33, color='black', linestyle='dashed')
    ax.legend(['Glucosa = 120', 'Edad = 33', 'Sano', 'Enfermo'])
    st.pyplot(fig)

    I = (df['Glucose'] <= 120).sum()
    D = (df['Glucose'] > 120).sum()
    PI = I / (I + D)
    PD = D / (I + D)

    sanos_I = ((df['Glucose'] <= 120) & (df['Outcome'] == 0)).sum()
    enfermos_I = ((df['Glucose'] <= 120) & (df['Outcome'] == 1)).sum()
    sanos_D = ((df['Glucose'] > 120) & (df['Outcome'] == 0)).sum()
    enfermos_D = ((df['Glucose'] > 120) & (df['Outcome'] == 1)).sum()

    def Gini(C1, C2):
        return 2 * (C1 / (C1 + C2)) * (C2 / (C1 + C2))

    def Gini_pond(GI, GD, PI, PD):
        return PI * GI + PD * GD

    GiniI = Gini(sanos_I, enfermos_I)
    GiniD = Gini(sanos_D, enfermos_D)
    Gini_ini = Gini(PI, PD)
    Coste = Gini_pond(GiniI, GiniD, PI, PD)
    Ganancia = Gini_ini - Coste

    st.markdown(f"""
    - **Gini Izquierda:** {GiniI:.2f}  
    - **Gini Derecha:** {GiniD:.2f}  
    - **Gini Inicial:** {Gini_ini:.2f}  
    - **Función de Coste (Gini ponderado):** {Coste:.2f}  
    - **Ganancia de Información:** {Ganancia:.2f}
    """)

    st.subheader("Visualización del Árbol")
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X, y)
    fig2, ax2 = plt.subplots(figsize=(6, 8))
    plot_tree(tree, max_depth=2, filled=True, feature_names=['Glucose', 'Age'], class_names=['Sano', 'Enfermo'], ax=ax2)
    st.pyplot(fig2)

# --- Página principal con métricas, predicción y visualizaciones ---
def pagina_clasificador():
    df = carga_datos()
    max_depth, criterion, edad, peso, volumen_o2 = add_sidebar(df)
    model, X_test, y_test = entrena_modelo(df, max_depth, criterion)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    pagina = st.sidebar.radio("Selecciona una página:", ['Métricas y Predicción', 'Gráficos'])

    if pagina == 'Métricas y Predicción':
        st.title('Métricas del Modelo y Predicción')
        col1, col2 = st.columns(2)
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
        st.title('Gráficos del Modelo')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Matriz de Confusión')
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', ax=ax)
            st.pyplot(fig)

            st.subheader('Curva ROC-AUC')
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], linestyle='--')
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader('Distribución de Predicciones')
            fig, ax = plt.subplots()
            pd.Series(y_pred).value_counts().plot(kind='bar', ax=ax, color=['blue', 'green'])
            ax.set_xticklabels(['Velocista', 'Fondista'], rotation=0)
            st.pyplot(fig)

            st.subheader('Árbol de Decisión')
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_tree(model, filled=True, feature_names=['Edad', 'Peso', 'Volumen_O2_max'], class_names=['Velocista', 'Fondista'], ax=ax)
            st.pyplot(fig)

# --- Inicio ---
def main():
    pagina = st.sidebar.selectbox("Selecciona una sección", ["Clasificador de Atletas", "Árbol y Gini Manual"])
    if pagina == "Clasificador de Atletas":
        pagina_clasificador()
    elif pagina == "Árbol y Gini Manual":
        pagina_arbol_manual()

if __name__ == '__main__':
    main()
