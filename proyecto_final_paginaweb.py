# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:18:03 2025
@author: zegar
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Título
st.title("Predicción de Satisfacción de Vida")

# Cargar datos con caché
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_estadistica.csv")

ds = cargar_datos()

# Mostrar primeras filas
st.write("Vista previa de los datos")
st.dataframe(ds.head())

# Visualización
st.subheader("Relaciones entre variables")
fig1 = sns.pairplot(ds)
st.pyplot(fig1)

# Matriz de correlación
st.subheader("Matriz de correlación")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(ds.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Variables predictoras y objetivo
X = ds[['Edad', 'Ingreso_Mensual', 'Hrs_Estudio_Semanal']]
y = ds['Satisfaccion_Vida']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Selector de modelo
st.sidebar.header("1. Selecciona el Modelo de Predicción")
modelo_seleccionado = st.sidebar.selectbox("Modelo", ["Regresión Lineal", "Árbol de Decisión", "KNN"])

# Inicializar modelo según selección
if modelo_seleccionado == "Regresión Lineal":
    modelo = LinearRegression()
elif modelo_seleccionado == "Árbol de Decisión":
    modelo = DecisionTreeRegressor()
elif modelo_seleccionado == "KNN":
    modelo = KNeighborsRegressor(n_neighbors=5)

# Entrenar el modelo
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Evaluación
st.subheader("Evaluación del Modelo Seleccionado")
st.write("Modelo:", modelo_seleccionado)
st.write("MSE:", round(mean_squared_error(y_test, y_pred), 2))
st.write("R²:", round(r2_score(y_test, y_pred), 2))

# Comparación gráfica
st.subheader("Comparación: Valores Reales vs Predichos")
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred, alpha=0.6)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax3.set_xlabel("Real")
ax3.set_ylabel("Predicho")
ax3.set_title("Comparación Real vs Predicción")
st.pyplot(fig3)

# Formulario de predicción individual
st.sidebar.header("2. Predecir nivel de satisfacción")
Edad = st.sidebar.slider("Edad", 0, 20, 10)
Ingreso = st.sidebar.slider("Ingreso Mensual (0-10)", 0, 10, 5)
Hrs_Estudio = st.sidebar.slider("Horas de Estudio Semanales", 0, 20, 10)

if st.sidebar.button("Predecir nivel de satisfacción"):
    pred_nueva = modelo.predict([[Edad, Ingreso, Hrs_Estudio]])
    st.sidebar.success(f"Predicción: {pred_nueva[0]:.2f}")

