# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:18:03 2025

@author: zegar
"""

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Título en la app
st.title("Base de Datos de Predección de Satisfacción de Vida")

# Función para cargar datos con caché
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_estadistica.csv")

# Cargar datos
ds = cargar_datos()

# Mostrar primeras filas
st.write("Vista previa de los datos")
st.dataframe(ds.head())



