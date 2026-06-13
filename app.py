"""
Aplicación de Detección de Exoplanetas - ExoIA
Integrantes: Jefferson Arce, Brayan Erazo, Kevin Grisales Rodriguez
Docente: Edgar Leonairo Pencue
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 0. CONFIGURACIÓN DE PÁGINA (DEBE IR PRIMERO)
# ============================================
st.set_page_config(
    page_title="ExoIA - Detector de Exoplanetas",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# 0b. IMÁGENES EN BASE64 (FONDO Y LOGO)
# ============================================
import base64

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Intentar cargar fondo y logo embebidos
_bg_paths = [Path("background.jpg"), Path("assets/background.jpg")]
_logo_paths = [Path("logo.png"), Path("assets/logo.png")]

_bg_b64 = ""
for p in _bg_paths:
    if p.exists():
        _bg_b64 = img_to_base64(str(p))
        break

_logo_b64 = ""
_logo_ext = "png"
for p in _logo_paths:
    if p.exists():
        _logo_b64 = img_to_base64(str(p))
        _logo_ext = p.suffix.lstrip(".")
        break

# ============================================
# 0c. ESTILOS GLOBALES
# ============================================
_bg_css = f"background-image: url('data:image/jpeg;base64,{_bg_b64}');" if _bg_b64 else ""

st.markdown(f"""
<style>
    /* Ocultar sidebar completamente */
    [data-testid="stSidebar"] {{display: none !important;}}
    [data-testid="collapsedControl"] {{display: none !important;}}

    /* Fondo de pantalla */
    .stApp {{
        {_bg_css}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Overlay semitransparente para legibilidad */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(5, 10, 25, 0.55);
        z-index: 0;
        pointer-events: none;
    }}

    /* Header con logo + título */
    .exoia-header {{
        display: flex;
        align-items: center;
        gap: 30px;
        padding: 10px 0 6px 0;
    }}
    .exoia-header img {{
        height: 190px;
        width: 190px;
    }}
    .exoia-header-text h1 {{
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 12px rgba(0,255,180,0.4);
    }}
    .exoia-header-text p {{
        margin: 4px 0 0 0;
        color: #a0f0d0;
        font-size: 0.95rem;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================
# 1. CARGA DEL MODELO
# ============================================

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado desde el archivo"""
    # Buscar en múltiples rutas posibles
    rutas = [
        Path('models/modelo_exoplanetas.pkl'),
        Path('modelo_exoplanetas.pkl'),
        Path('../modelo_exoplanetas.pkl'),
        Path('models/modelo_exoplanetas.joblib'),
        Path('modelo_exoplanetas.joblib')
    ]
    
    for ruta in rutas:
        if ruta.exists():
            try:
                modelo = joblib.load(ruta)
                st.success(f"✅ Modelo cargado desde: {ruta.name}")
                return modelo
            except Exception as e:
                st.warning(f"Error al cargar {ruta.name}: {e}")
                continue
    
    st.error("❌ No se encontró el modelo entrenado")
    st.info("""
    **Solución:** 
    1. Coloca el archivo `modelo_exoplanetas.pkl` en la carpeta `models/`
    2. O en la misma carpeta que `app.py`
    3. Reinicia la aplicación
    """)
    return None

@st.cache_resource
def cargar_label_encoder():
    """Carga el label encoder"""
    rutas = [
        Path('models/label_encoder.pkl'),
        Path('label_encoder.pkl'),
        Path('../label_encoder.pkl')
    ]
    
    for ruta in rutas:
        if ruta.exists():
            try:
                le = joblib.load(ruta)
                st.success(f"✅ Label encoder cargado")
                return le
            except:
                continue
    return None

# ============================================
# 2. CARGA DE BASE DE DATOS
# ============================================

@st.cache_data
def cargar_datos_ejemplo():
    """Datos de ejemplo para demostración (exoplanetas confirmados)"""
    return pd.DataFrame({
        'kepoi_name': ['K00752.01', 'K00755.01', 'K00756.01', 'K00756.02', 'K00756.03', 
                       'K00753.01', 'K00754.01', 'K00757.01', 'K00001.01', 'K00002.01'],
        'kepler_name': ['Kepler-227 b', 'Kepler-664 b', 'Kepler-228 d', 'Kepler-228 c', 
                        'Kepler-228 b', 'FALSE POSITIVE', 'FALSE POSITIVE', 'Kepler-229 c',
                        'Kepler-1 b', 'Kepler-2 b'],
        'kepid': [10797460, 10854555, 10872983, 10872983, 10872983, 10811496, 10884559, 10910878, 11446443, 10666592],
        'koi_pdisposition': ['CANDIDATE', 'CANDIDATE', 'CANDIDATE', 'CANDIDATE', 'CANDIDATE',
                            'FALSE POSITIVE', 'FALSE POSITIVE', 'CANDIDATE', 'CANDIDATE', 'CANDIDATE'],
        'koi_period': [9.488036, 2.525592, 11.094321, 4.134435, 2.566589, 19.899140, 1.736952, 16.063647, 2.470613, 2.204735],
        'koi_prad': [2.26, 2.75, 3.90, 2.77, 1.59, 14.60, 33.46, 5.76, 1.00, 1.00],
        'koi_teq': [793.0, 1406.0, 835.0, 1160.0, 1360.0, 638.0, 1395.0, 600.0, 1000.0, 1000.0],
        'koi_duration': [2.95750, 1.65450, 4.59450, 3.14020, 2.42900, 1.78220, 2.40641, 3.53470, 2.50000, 2.50000],
        'koi_depth': [616.0, 603.0, 1520.0, 686.0, 227.0, 10800.0, 8080.0, 4910.0, 500.0, 500.0],
        'koi_model_snr': [35.8, 40.9, 66.5, 40.2, 15.0, 76.3, 505.6, 161.9, 30.0, 30.0],
        'koi_steff': [5455.0, 6031.0, 6046.0, 6046.0, 6046.0, 5853.0, 5805.0, 5031.0, 5500.0, 5500.0],
        'koi_slogg': [4.467, 4.438, 4.486, 4.486, 4.486, 4.544, 4.564, 4.485, 4.500, 4.500],
        'koi_srad': [0.927, 1.046, 0.972, 0.972, 0.972, 0.868, 0.791, 0.848, 1.000, 1.000],
        'koi_impact': [0.146, 0.701, 0.538, 0.762, 0.755, 0.969, 1.276, 0.052, 0.500, 0.500],
        'koi_time0bk': [170.53875, 171.59555, 171.20116, 172.97937, 179.55437, 175.850252, 170.307565, 173.621937, 170.000, 170.000],
        'koi_insol': [93.59, 926.16, 114.81, 427.65, 807.74, 39.30, 891.96, 30.75, 100.00, 100.00],
        'koi_kepmag': [15.347, 15.509, 15.714, 15.714, 15.714, 15.436, 15.597, 15.841, 14.000, 14.000],
        'koi_tce_plnt_num': [1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })

@st.cache_data
def cargar_base_datos():
    """Carga la base de datos real de exoplanetas"""
    
    # Opción 1: Archivo local en la misma carpeta
    ruta_local = Path('baseDatosExoplanetas.csv')
    if ruta_local.exists():
        df = pd.read_csv(ruta_local)
        st.success(f"✅ Base de datos cargada: {len(df)} registros")
        return df
    
    # Opción 2: Carpeta data/
    ruta_data = Path('data/baseDatosExoplanetas.csv')
    if ruta_data.exists():
        df = pd.read_csv(ruta_data)
        st.success(f"✅ Base de datos cargada: {len(df)} registros")
        return df
    
    # Opción 3: Carpeta Bases de datos/
    ruta_bd = Path('Bases de datos/baseDatosExoplanetas.csv')
    if ruta_bd.exists():
        df = pd.read_csv(ruta_bd)
        st.success(f"✅ Base de datos cargada: {len(df)} registros")
        return df
    
    # Si no hay archivo, usar datos de ejemplo
    st.info("📋 Usando base de datos de ejemplo (exoplanetas confirmados).")
    st.info("Para usar datos reales, coloca 'baseDatosExoplanetas.csv' en la carpeta del proyecto.")
    return cargar_datos_ejemplo()

# ============================================
# 3. FUNCIONES AUXILIARES
# ============================================

def buscar_por_identificador(identificador, df):
    """Busca un exoplaneta por su kepid, kepoi_name o kepler_name"""
    if df is None or df.empty or not identificador:
        return pd.DataFrame()
    
    identificador = str(identificador).strip()
    
    # Buscar en diferentes columnas
    resultado = df[
        (df['kepoi_name'].astype(str).str.contains(identificador, case=False, na=False)) |
        (df['kepler_name'].astype(str).str.contains(identificador, case=False, na=False)) |
        (df['kepid'].astype(str).str.contains(identificador, case=False, na=False))
    ]
    
    return resultado

def safe_format(value, format_str="{:.4f}", default="N/A"):
    """Formatea valores seguros manejando NaN"""
    if pd.isna(value):
        return default
    try:
        return format_str.format(value)
    except (TypeError, ValueError):
        return str(value)

def predecir_exoplaneta(registro, modelo, le):
    """Predice si un registro corresponde a un exoplaneta verdadero"""
    
    # Verificar si el modelo está cargado
    if modelo is None:
        st.warning("⚠️ **MODELO NO CARGADO** - Usando modo demostración")
        
        # Modo demostración: predecir por nombre
        nombre = str(registro.get('kepler_name', ''))
        es_exoplaneta = 'false positive' not in nombre.lower() and nombre not in ['', 'nan', 'None']
        return {
            'etiqueta': 'CANDIDATE' if es_exoplaneta else 'FALSE POSITIVE',
            'prob_candidato': 0.85 if es_exoplaneta else 0.15,
            'prob_falso': 0.15 if es_exoplaneta else 0.85,
            'es_exoplaneta': es_exoplaneta
        }
    
    # Predicción real con el modelo
    try:
        # Columnas necesarias para el modelo
        columnas_features = ['koi_srad', 'koi_period', 'koi_time0bk', 'koi_impact', 
                             'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 
                             'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_slogg',
                             'koi_kepmag', 'koi_tce_plnt_num']
        
        # Crear DataFrame con los valores
        valores = {}
        for col in columnas_features:
            val = registro.get(col, 0)
            # Manejar NaN
            if pd.isna(val):
                val = 0
            valores[col] = val
        
        X = pd.DataFrame([valores])
        
        # Realizar predicción
        proba = modelo.predict_proba(X)[0]
        prediccion = modelo.predict(X)[0]
        
        # Determinar etiqueta
        if le is not None:
            try:
                etiqueta = le.inverse_transform([prediccion])[0]
            except:
                etiqueta = 'CANDIDATE' if prediccion == 1 else 'FALSE POSITIVE'
        else:
            etiqueta = 'CANDIDATE' if prediccion == 1 else 'FALSE POSITIVE'
        
        return {
            'etiqueta': etiqueta,
            'prob_candidato': float(proba[1]) if len(proba) > 1 else 0.5,
            'prob_falso': float(proba[0]) if len(proba) > 0 else 0.5,
            'es_exoplaneta': (prediccion == 1) or (etiqueta == 'CANDIDATE')
        }
        
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return {
            'etiqueta': 'ERROR',
            'prob_candidato': 0.5,
            'prob_falso': 0.5,
            'es_exoplaneta': False
        }

# ============================================
# 4. INTERFAZ PRINCIPAL
# ============================================

# Header: logo + título al lado
_logo_tag = f'<img src="data:image/{_logo_ext};base64,{_logo_b64}" />' if _logo_b64 else "🪐"

st.markdown(f"""
<div class="exoia-header">
    {_logo_tag}
    <div class="exoia-header-text">
        <h1 style="font-size: 48px;">ExoIA - Sistema de Detección de Exoplanetas</h1>
        <p style="font-size: 20px;">Aplicación basada en Machine Learning para la identificación de exoplanetas a partir de datos del telescopio Kepler.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Cargar recursos
with st.spinner("Cargando recursos..."):
    modelo = cargar_modelo()
    le = cargar_label_encoder()
    datos = cargar_base_datos()

# ============================================
# 5. BÚSQUEDA
# ============================================

st.header("🪐 Buscar exoplaneta:")

col1, col2 = st.columns([3, 1])
with col1:
    identificador = st.text_input(
        "Ingresa el identificador:",
        placeholder="Ejemplo: K00752.01, Kepler-227, 10797460",
        help="Puedes usar kepid, kepoi_name o kepler_name",
        label_visibility="collapsed"
    )

with col2:
    buscar = st.button("🔍 Buscar", type="primary", width='stretch')

if buscar and identificador:
    with st.spinner("Buscando..."):
        resultado = buscar_por_identificador(identificador, datos)
        
        if resultado.empty:
            st.error(f"❌ No se encontró ningún registro con: '{identificador}'")
            
            st.info("💡 **Sugerencias de búsqueda válidas:**")
            ejemplos = datos['kepoi_name'].dropna().head(5).tolist()
            for ej in ejemplos:
                st.write(f"   - `{ej}`")
        else:
            st.success(f"✅ Se encontró {len(resultado)} registro(s)")
            
            for idx, (_, registro) in enumerate(resultado.iterrows()):
                if idx > 0:
                    st.markdown("---")
                
                # Realizar predicción
                prediccion = predecir_exoplaneta(registro, modelo, le)
                
                col_info, col_pred = st.columns([2, 1])
                
                with col_info:
                    kepoi = registro.get('kepoi_name', 'Desconocido')
                    kepler = registro.get('kepler_name', '')
                    titulo = f"📋 {kepoi}"
                    if kepler and kepler != 'FALSE POSITIVE' and not pd.isna(kepler):
                        titulo += f" - {kepler}"
                    st.subheader(titulo)
                    
                    # Construir diccionario de información
                    info = {
                        "ID Kepler": str(registro.get('kepid', 'N/A')),
                        "Nombre KOI": str(registro.get('kepoi_name', 'N/A')),
                        "Nombre Kepler": str(registro.get('kepler_name', 'N/A')) if pd.notna(registro.get('kepler_name')) else "N/A",
                        "Período (días)": safe_format(registro.get('koi_period'), "{:.4f}"),
                        "Radio (R_Tierra)": safe_format(registro.get('koi_prad'), "{:.2f}"),
                        "Temperatura (K)": safe_format(registro.get('koi_teq'), "{:.0f}"),
                        "Duración (horas)": safe_format(registro.get('koi_duration'), "{:.2f}"),
                        "Profundidad (ppm)": safe_format(registro.get('koi_depth'), "{:.0f}"),
                        "SNR": safe_format(registro.get('koi_model_snr'), "{:.1f}"),
                        "Temperatura Estrella (K)": safe_format(registro.get('koi_steff'), "{:.0f}"),
                        "log(g)": safe_format(registro.get('koi_slogg'), "{:.3f}"),
                        "Insolación": safe_format(registro.get('koi_insol'), "{:.2f}"),
                        "Magnitud Kepler": safe_format(registro.get('koi_kepmag'), "{:.3f}"),
                        "Número de planetas": safe_format(registro.get('koi_tce_plnt_num'), "{:.0f}")
                    }
                    
                    # Crear DataFrame y asegurar tipos string
                    df_info = pd.DataFrame(info.items(), columns=["Parámetro", "Valor"])
                    df_info["Valor"] = df_info["Valor"].astype(str)
                    
                    st.dataframe(df_info, width='stretch', hide_index=True)
                
                with col_pred:
                    st.subheader("📊 Predicción")
                    
                    st.metric("Confianza FALSE POSITIVE", f"{prediccion['prob_candidato']*100:.1f}%")
                    st.metric("Confianza CANDIDATE", f"{prediccion['prob_falso']*100:.1f}%")
                    
                    # Mostrar advertencia si está en modo demostración
                    if modelo is None:
                        st.warning("⚠️ Modo demostración - Carga el modelo para predicciones reales")

# ============================================
# 6. SECCIÓN DE PRUEBA MANUAL
# ============================================

st.header("⚙️ Prueba personalizada")

with st.expander("Ingresar valores manuales para predicción"):
    st.markdown("Ingresa los parámetros observacionales para clasificar si es exoplaneta:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        periodo = st.number_input("Período orbital (días)", value=10.0, step=1.0, key="periodo_input")
        duracion = st.number_input("Duración del tránsito (horas)", value=3.0, step=0.5, key="duracion_input")
        profundidad = st.number_input("Profundidad (ppm)", value=500.0, step=100.0, key="profundidad_input")
    
    with col2:
        prad = st.number_input("Radio planetario (R_Tierra)", value=2.0, step=0.5, key="prad_input")
        teq = st.number_input("Temperatura equilibrio (K)", value=800.0, step=50.0, key="teq_input")
        insol = st.number_input("Insolación", value=100.0, step=50.0, key="insol_input")
    
    with col3:
        snr = st.number_input("SNR", value=30.0, step=10.0, key="snr_input")
        steff = st.number_input("Temperatura estrella (K)", value=5500.0, step=100.0, key="steff_input")
        slogg = st.number_input("log(g)", value=4.5, step=0.1, key="slogg_input")
    
    if st.button("Clasificar", type="primary", key="clasificar_btn"):
        # Simulación de predicción basada en parámetros
        es_exoplaneta_sim = (periodo > 2.5 and prad > 1.5 and snr > 20 and prad < 20)
        
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            st.markdown("---")
            if es_exoplaneta_sim:
                st.success(f"🪐 **PREDICCIÓN: EXOPLANETA**")
                confianza = min(95, 50 + periodo + prad*5)
                st.metric("Confianza estimada", f"{confianza:.0f}%")
                st.info("💡 *Simulación - Para predicción real usa el modelo entrenado.*")
            else:
                st.error(f"❌ **PREDICCIÓN: FALSO POSITIVO**")
                confianza = min(90, 40 + snr/2)
                st.metric("Confianza estimada", f"{confianza:.0f}%")
                st.info("💡 *Simulación - Para predicción real usa el modelo entrenado.*")

# ============================================
# 7. PIE DE PÁGINA
# ============================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white;'>
        <p>Universidad del Cauca | Ingeniería Física</p>
        <p>ExoIA - Detección de Exoplanetas | Procesamiento de Señales</p>
        <p>Jefferson Arce Eraso, Brayan Erazo, Kevin Grisales Rodriguez</p>
    </div>
    """,
    unsafe_allow_html=True
)