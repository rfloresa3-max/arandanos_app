import streamlit as st
from pathlib import Path
import os
from helpers import load_model, procesar_imagen, load_classes_from_yaml

# ----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------
st.set_page_config(page_title="Monitor de Arándanos", layout="wide")
st.title("🫐 Monitor de Inspección - Arándanos")
st.markdown("Sube una imagen para detectar y clasificar arándanos por estado de maduración.")

# ----------------------------
# RUTAS Y CONFIGURACIÓN
# ----------------------------
MODEL_PATH = "models/student_blueberry.pt"
YAML_PATH = "blueberry-images-seg-1/dataset.yaml"

days_map = {
    'mature': 0,
    'overmature': 3,
    'semi-mature': -14,
    'immature': -21
}

# ----------------------------
# CARGA DE MODELO Y CLASES
# ----------------------------
try:
    model = load_model(MODEL_PATH)
    class_names = load_classes_from_yaml(YAML_PATH)
    st.success(f"✅ Modelo cargado correctamente. Clases: {class_names}")
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# ----------------------------
# INTERFAZ DE USUARIO
# ----------------------------
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = Path("temp_image.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Procesando imagen..."):
        img_rgb, panel_text, error = procesar_imagen(temp_path, model, class_names, days_map)
    
    if error:
        st.error(error)
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img_rgb, caption="Detecciones", use_container_width=True)
        with col2:
            # Panel no editable y con estilo nativo de Streamlit
            st.code(panel_text, language='text')
    
    if temp_path.exists():
        temp_path.unlink()
else:
    st.info("Sube una imagen para comenzar el análisis.")