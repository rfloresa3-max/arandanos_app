import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
import streamlit as st

# ----------------------------
# CARGA DE MODELO (con caché)
# ----------------------------
@st.cache_resource
def load_model(model_path):
    """
    Carga el modelo YOLO desde la ruta especificada.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
    model = YOLO(model_path)
    return model

# ----------------------------
# CARGA DE CLASES DESDE YAML
# ----------------------------
def load_classes_from_yaml(yaml_path):
    """
    Lee el archivo dataset.yaml y devuelve la lista de nombres de clases.
    Si no encuentra el archivo, usa valores por defecto.
    """
    if Path(yaml_path).exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        class_names = data.get('names', ['immature', 'mature', 'overmature', 'semi-mature'])
    else:
        # Fallback si no existe el YAML
        class_names = ['immature', 'mature', 'overmature', 'semi-mature']
    return class_names

# ----------------------------
# PROCESAMIENTO DE IMAGEN
# ----------------------------
def procesar_imagen(image_path, model, class_names, days_map):
    """
    Procesa una imagen, ejecuta el modelo YOLO, dibuja bounding boxes
    y genera el texto del panel de inspección.
    """
    # Cargar imagen
    img_original = cv2.imread(str(image_path))
    if img_original is None:
        return None, None, "Error: No se pudo cargar la imagen"
    
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    # Predicción
    results = model.predict(str(image_path), imgsz=640, conf=0.25, iou=0.5)[0]
    
    # Conteo
    counts = {name: 0 for name in class_names}
    confidences = []
    detections = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls < len(class_names):
            name = class_names[cls]
            counts[name] += 1
            conf = float(box.conf[0])
            confidences.append(conf)
            detections.append((name, conf, box.xyxy[0].tolist()))
    
    total = sum(counts.values())
    avg_conf = np.mean(confidences) if confidences else 0.0
    
    # Dibujar bounding boxes
    for name, conf, (x1, y1, x2, y2) in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{name} {conf:.2f}"
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Construir texto del panel
    panel_lines = []
    panel_lines.append("=== MONITOR DE INSPECCION ===")
    panel_lines.append(f"Total objetos: {total}")
    panel_lines.append(f"Confianza promedio: {avg_conf:.2f}")
    panel_lines.append("")
    panel_lines.append("--- Conteo por clase ---")
    for name in class_names:
        count = counts[name]
        days = days_map.get(name, "?")
        if days == 0:
            days_str = "0"
        elif days > 0:
            days_str = f"+{days}"
        else:
            days_str = f"{days}"
        panel_lines.append(f"{name}: {count} (dias: {days_str})")
    panel_lines.append("")
    panel_lines.append("--- Veredicto ---")
    if counts.get('mature', 0) >= (total * 0.5) and total > 0:
        panel_lines.append("APTO para cosecha (mayoria madura)")
    else:
        panel_lines.append("REVISAR (esperar maduracion)")
    
    panel_text = "\n".join(panel_lines)
    
    return img_rgb, panel_text, None