# -*- coding: utf-8 -*-
"""
Predicción de Satisfacción de Vida con diferentes modelos
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configuración de página
st.set_page_config(page_title="Predicción de Satisfacción de Vida", layout="wide")

# Título principal
st.title("🔍 Predicción de Satisfacción de Vida")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_estadistica.csv")

ds = cargar_datos()

# Mostrar columnas
st.write("📋 Columnas del dataset:", ds.columns.tolist())

# Vista previa
st.subheader("📌 Vista previa de los datos")
st.dataframe(ds.head())

# Estadística descriptiva
st.subheader("📊 Estadística Descriptiva de Variables Numéricas")
st.dataframe(ds.describe())

# Variables numéricas
ds_num = ds.select_dtypes(include='number')

# Pairplot
st.subheader("📊 Relaciones entre variables numéricas (Pairplot)")
fig1 = sns.pairplot(ds_num)
st.pyplot(fig1)

# Matriz de correlación
st.subheader("📈 Matriz de correlación")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(ds_num.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Histogramas + KDE
st.subheader("📉 Distribuciones de variables numéricas")
for columna in ds_num.columns:
    fig, ax = plt.subplots()
    sns.histplot(ds[columna], kde=True, ax=ax, color='skyblue')
    ax.set_title(f"Distribución de {columna}")
    st.pyplot(fig)

# Boxplots
st.subheader("📦 Boxplots de variables numéricas")
for columna in ds_num.columns:
    fig, ax = plt.subplots()
    sns.boxplot(x=ds[columna], ax=ax, color='lightgreen')
    ax.set_title(f"Boxplot de {columna}")
    st.pyplot(fig)

# Validación de columnas
columnas_requeridas = ['Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal', 'Satisfaccion_Vida']
if all(col in ds.columns for col in columnas_requeridas):

    # Selección de variables
    X = ds[['Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal']]
    y = ds['Satisfaccion_Vida']

    # Limpieza
    df_limpio = pd.concat([X, y], axis=1)
    df_limpio.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_limpio.dropna(inplace=True)

    X = df_limpio[['Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal']]
    y = df_limpio['Satisfaccion_Vida']

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sidebar: Selección de modelo
    st.sidebar.header("1️⃣ Selecciona el modelo de predicción")
    modelo_seleccionado = st.sidebar.selectbox("Modelo", ["Regresión Lineal", "Árbol de Decisión", "KNN"])

    # Normalización solo para KNN
    if modelo_seleccionado == "KNN":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        normalizador_aplicado = True
    else:
        normalizador_aplicado = False

    # Crear modelo
    if modelo_seleccionado == "Regresión Lineal":
        modelo = LinearRegression()
    elif modelo_seleccionado == "Árbol de Decisión":
        modelo = DecisionTreeRegressor()
    elif modelo_seleccionado == "KNN":
        modelo = KNeighborsRegressor(n_neighbors=5)

    # Entrenamiento
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Resultados
    st.subheader("📋 Evaluación del modelo")
    st.write(f"🔧 Modelo seleccionado: **{modelo_seleccionado}**")
    if normalizador_aplicado:
        st.info("⚙️ Se aplicó normalización estándar porque se eligió KNN.")
    st.write("✅ Error Cuadrático Medio (MSE):", round(mean_squared_error(y_test, y_pred), 2))
    st.write("✅ Coeficiente de Determinación (R²):", round(r2_score(y_test, y_pred), 2))

    # Gráfico de comparación
    st.subheader("📌 Comparación: Valor real vs predicho")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred, alpha=0.6)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Predicho")
    ax3.set_title("Comparación Real vs Predicción")
    st.pyplot(fig3)

    # Formulario individual
    st.sidebar.header("2️⃣ Predecir nueva satisfacción de vida")
    Edad = st.sidebar.slider("Edad", 18, 80, 30)
    Ingreso = st.sidebar.slider("Ingreso Mensual (Bs)", 0, 10000, 3000)
    Hrs_Estudio = st.sidebar.slider("Horas de estudio semanales", 0, 40, 10)

    if st.sidebar.button("🔮 Predecir"):
        entrada = np.array([[Edad, Ingreso, Hrs_Estudio]])
        if normalizador_aplicado:
            entrada = scaler.transform(entrada)
        pred_nueva = modelo.predict(entrada)
        st.sidebar.success(f"Nivel de satisfacción estimado: {pred_nueva[0]:.2f}")

else:
    st.error("❌ Las columnas necesarias no existen en el dataset. Revisa que tengas: 'Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal', 'Satisfaccion_Vida'")
