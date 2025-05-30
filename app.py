import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.model_selection import train_test_split

# Configuración de la página
st.set_page_config(page_title='Clasificador de Atletas', layout='wide')

# Cargar datos
@st.cache_data
def carga_datos():
    df = pd.read_csv('atletas.csv')
    df['Atleta'] = df['Atleta'].astype(str).str.strip().str.lower()
    df['Atleta'] = df['Atleta'].map({'fondista': 1, 'velocista': 0})
    df.dropna(inplace=True)
    return df

# Sidebar con controles
def add_sidebar(df):
    st.sidebar.header('Configuración del Modelo')
    
    max_depth = st.sidebar.slider('Profundidad máxima del árbol', 2, 10, 3)
    criterion = st.sidebar.selectbox('Criterio de división', ['gini', 'entropy'])
    
    st.sidebar.subheader('Variables del Modelo')
    available_features = ['Edad', 'Peso', 'Altura', 'Volumen_O2_max', 'Umbral_lactato', '%Fibras_rapidas', '%Fibras_lentas']
    available_in_df = [col for col in available_features if col in df.columns]
    
    selected_features = st.sidebar.multiselect(
        'Selecciona las variables a usar:',
        available_in_df,
        default=available_in_df[:3] if len(available_in_df) >= 3 else available_in_df
    )
    
    if not selected_features:
        st.sidebar.error("Selecciona al menos una variable")
        selected_features = available_in_df[:1]  # Selecciona una por defecto
    
    st.sidebar.subheader('Datos del Nuevo Atleta para Predicción')
    
    atleta_data = {}
    for feature in selected_features:
        if feature == 'Edad':
            atleta_data['Edad'] = st.sidebar.number_input("Edad", min_value=15, max_value=80, value=25)
        elif feature == 'Peso':
            atleta_data['Peso'] = st.sidebar.number_input("Peso (kg)", min_value=40.0, max_value=120.0, value=70.0)
        elif feature == 'Altura':
            atleta_data['Altura'] = st.sidebar.number_input("Altura (cm)", min_value=140.0, max_value=220.0, value=175.0)
        elif feature == 'Volumen_O2_max':
            atleta_data['Volumen_O2_max'] = st.sidebar.number_input("Volumen O2 máx", min_value=20.0, max_value=90.0, value=50.0)
        elif feature == 'Umbral_lactato':
            atleta_data['Umbral_lactato'] = st.sidebar.number_input("Umbral de lactato", min_value=2.0, max_value=6.0, value=4.0)
        elif feature == '%Fibras_rapidas':
            atleta_data['%Fibras_rapidas'] = st.sidebar.slider("% Fibras rápidas", 0, 100, 50)
            if '%Fibras_lentas' in selected_features:
                atleta_data['%Fibras_lentas'] = 100 - atleta_data['%Fibras_rapidas']
                st.sidebar.write(f"% Fibras lentas: {atleta_data['%Fibras_lentas']}")
        elif feature == '%Fibras_lentas' and '%Fibras_rapidas' not in selected_features:
            atleta_data['%Fibras_lentas'] = st.sidebar.slider("% Fibras lentas", 0, 100, 50)
    
    return max_depth, criterion, selected_features, atleta_data

# Entrenamiento del modelo
def entrenar_modelo(df, max_depth, criterion, selected_features):
    X = df[selected_features]
    y = df['Atleta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree_model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)
    tree_prob = tree_model.predict_proba(X_test)[:, 1]
    
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)
    log_prob = log_model.predict_proba(X_test)[:, 1]
    
    return {
        'X_test': X_test, 'y_test': y_test,
        'tree_model': tree_model, 'log_model': log_model,
        'tree_pred': tree_pred, 'log_pred': log_pred,
        'tree_prob': tree_prob, 'log_prob': log_prob,
        'feature_columns': selected_features
    }

# Función principal
def main():
    st.title('🏃‍♂️ Clasificador de Atletas')
    
    try:
        df = carga_datos()
    except:
        st.error("No se pudo cargar el archivo 'atletas.csv'. Asegúrate de que existe.")
        return
    
    if df.empty:
        st.error("No hay datos válidos en el archivo CSV.")
        return
    
    max_depth, criterion, selected_features, atleta_data = add_sidebar(df)
    resultados = entrenar_modelo(df, max_depth, criterion, selected_features)
    
    # Predicción del nuevo atleta
    if atleta_data:
        nuevo_atleta = pd.DataFrame([atleta_data])
        prediccion = resultados['tree_model'].predict(nuevo_atleta)[0]
        probabilidad = resultados['tree_model'].predict_proba(nuevo_atleta)[0]
        
        tipo = "Fondista" if prediccion == 1 else "Velocista"
        confianza = max(probabilidad) * 100
        
        st.success(f"**Predicción:** El atleta será un **{tipo}** (Confianza: {confianza:.1f}%)")
    
    tab1, tab2, tab3 = st.tabs(["📊 Métricas", "📈 Gráficos", "🔍 Comparación"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Métricas del Árbol de Decisión")
            accuracy = accuracy_score(resultados['y_test'], resultados['tree_pred'])
            st.metric("Precisión", f"{accuracy:.2f}")
            st.text("Reporte de Clasificación:")
            st.text(classification_report(resultados['y_test'], resultados['tree_pred']))
        with col2:
            st.subheader("Información del Dataset")
            st.write(f"Total de atletas: {len(df)}")
            st.write(f"Fondistas: {(df['Atleta'] == 1).sum()}")
            st.write(f"Velocistas: {(df['Atleta'] == 0).sum()}")
            st.write(f"Variables utilizadas: {len(resultados['feature_columns'])}")
            st.dataframe(df[resultados['feature_columns']].describe(), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Árbol de Decisión")
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(resultados['tree_model'], filled=True, feature_names=resultados['feature_columns'], class_names=["Velocista", "Fondista"], rounded=True, ax=ax)
            st.pyplot(fig)
            
            st.subheader("Matriz de Confusión")
            fig, ax = plt.subplots(figsize=(6, 4))
            conf_matrix = confusion_matrix(resultados['y_test'], resultados['tree_pred'])
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['Velocista', 'Fondista'], yticklabels=['Velocista', 'Fondista'], ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Curva ROC")
            fpr, tpr, _ = roc_curve(resultados['y_test'], resultados['tree_prob'])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.set_title('Curva ROC')
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("Distribución de Variables")
            for col in resultados['feature_columns']:
                if col in df.columns:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.histplot(data=df, x=col, hue='Atleta', bins=15, ax=ax)
                    ax.set_title(f'Distribución de {col}')
                    st.pyplot(fig)
    
    with tab3:
        st.subheader("Comparación: Árbol vs Regresión Logística")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Árbol de Decisión**")
            tree_acc = accuracy_score(resultados['y_test'], resultados['tree_pred'])
            tree_auc = roc_auc_score(resultados['y_test'], resultados['tree_prob'])
            st.metric("Precisión", f"{tree_acc:.3f}")
            st.metric("AUC", f"{tree_auc:.3f}")
        with col2:
            st.markdown("**Regresión Logística**")
            log_acc = accuracy_score(resultados['y_test'], resultados['log_pred'])
            log_auc = roc_auc_score(resultados['y_test'], resultados['log_prob'])
            st.metric("Precisión", f"{log_acc:.3f}")
            st.metric("AUC", f"{log_auc:.3f}")
        
        modelos = ['Árbol', 'Regresión']
        precisiones = [tree_acc, log_acc]
        aucs = [tree_auc, log_auc]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(modelos, precisiones, color=['skyblue', 'lightcoral'])
        ax1.set_title('Comparación de Precisión')
        ax1.set_ylabel('Precisión')
        ax2.bar(modelos, aucs, color=['skyblue', 'lightcoral'])
        ax2.set_title('Comparación de AUC')
        ax2.set_ylabel('AUC')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
