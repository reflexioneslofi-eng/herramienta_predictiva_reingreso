import streamlit as st
import pandas as pd
import joblib

# =========================================================
# CARGAR MODELO
# =========================================================
data = joblib.load("modelo_rf_final.pkl")
modelo = data["model"]

# =========================================================
# TÍTULO
# =========================================================
st.title("Predictor de Reingreso")

# =========================================================
# INPUTS
# =========================================================
estancia = st.number_input(
    "Estancia hospitalaria",
    min_value=0.0,
    max_value=60.0,
    value=7.0
)

rdw = st.number_input(
    "RDW",
    min_value=10.0,
    max_value=25.0,
    value=14.0
)

n_farmacos = st.number_input(
    "Número de fármacos",
    min_value=0,
    max_value=30,
    value=8
)

hb = st.number_input(
    "Hemoglobina preoperatoria",
    min_value=5.0,
    max_value=20.0,
    value=12.0
)

# --- Cambio aquí ---
asa_options = ["Desconocido", 2, 3, 4]
asa_display = st.selectbox(
    "ASA",
    asa_options
)
asa = 11 if asa_display == "Desconocido" else asa_display
# --- Fin del cambio ---

sexo = st.selectbox(
    "Sexo",
    ["Mujer", "Hombre"]
)
sexo = 1 if sexo == "Hombre" else 0

# =========================================================
# PREDICCIÓN
# =========================================================
if st.button("Predecir"):
    nuevo_paciente = pd.DataFrame({
        "estancia": [estancia],
        "pre_distribucion_eritrocitaria": [rdw],
        "n_farmacos_total": [n_farmacos],
        "pre_hb": [hb],
        "asa": [asa],
        "sexo": [sexo]
    })

    pred = modelo.predict(nuevo_paciente)[0]
    prob = modelo.predict_proba(nuevo_paciente)[0, 1]

    # =====================================================
    # RESULTADO
    # =====================================================
    if pred == 1:
        st.error(f"Riesgo de reingreso ({prob:.1%})")
    else:
        st.success(f"Bajo riesgo de reingreso ({prob:.1%})")
