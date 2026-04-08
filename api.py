import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. CONFIGURACIÓN VISUAL ROTOPLAS ---
st.set_page_config(page_title="Análisis Hidrología | Rotoplas", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F4F7F9; }
    [data-testid="stSidebar"] { background-color: #004C97; border-right: 3px solid #00A9E0; }
    [data-testid="stSidebar"] * { color: white !important; }
    h1 { color: #004C97; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    div[data-testid="stMetricValue"] { color: #00A9E0; }
    .stButton>button { background-color: #00A9E0; color: white; border-radius: 25px; width: 100%; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR Y FILTROS ---
with st.sidebar:
    # Intenta cargar el logo si existe, si no usa un título
    st.title("🚰 Rotoplas BI")
    st.markdown("---")
    st.subheader("🔑 Acceso")
    api_key = st.text_input("Introduce tu Google API Key", type="password")
    
    st.markdown("---")
    st.subheader("📅 Parámetros Globales")
    alcaldia_sel = st.selectbox("Alcaldía Prioritaria", ["Iztapalapa", "Benito Juárez", "Coyoacán", "Todas"])

# --- 3. CARGA Y UNIFICACIÓN DE LOS 4 ARCHIVOS ---
@st.cache_data
def load_full_dataset():
    # Lista exacta con tus nuevos nombres de archivo
    archivos = [
        "2025Q1.parquet",    # Enero - Marzo 2025
        "2025Q2P1.parquet",  # Abril - Mayo 2025
        "2025Q2P2.parquet",  # Mayo - Junio 2025
        "2025Q3P1.parquet",  # Julio - Agosto 2025
        "2025Q3P2.parquet",  # Agosto - Septiembre 2025
        "2026Q1.parquet"     # Enero - Marzo 2026
    ]
    
    list_dfs = []
    archivos_faltantes = []

    for f in archivos:
        if os.path.exists(f):
            try:
                temp_df = pd.read_parquet(f)
                list_dfs.append(temp_df)
            except Exception as e:
                st.sidebar.error(f"Error al leer {f}: {e}")
        else:
            archivos_faltantes.append(f)
    
    # Notificación en el sidebar si falta algo, pero permite continuar
    if archivos_faltantes:
        st.sidebar.warning(f"⚠️ Archivos no encontrados: {', '.join(archivos_faltantes)}")
    
    if not list_dfs:
        return pd.DataFrame()
    
    # Concatenar todos los archivos en uno solo
    df_unificado = pd.concat(list_dfs, ignore_index=True)
    
    # --- LIMPIEZA DE DATOS CRÍTICA ---
    # 1. Convertir fecha a datetime
    if 'fecha_lectura_time' in df_unificado.columns:
        df_unificado['fecha_lectura_time'] = pd.to_datetime(df_unificado['fecha_lectura_time'], errors='coerce')
        # Eliminar filas sin fecha válida si las hubiera
        df_unificado = df_unificado.dropna(subset=['fecha_lectura_time'])
        # Ordenar cronológicamente para que los gráficos salgan bien
        df_unificado = df_unificado.sort_values('fecha_lectura_time')
    
    # 2. Asegurar que los litros sean numéricos
    cols_numericas = ['volumen_actual_litros', 'capacidad_litros', 'porcentaje_nivel_liquido']
    for col in cols_numericas:
        if col in df_unificado.columns:
            df_unificado[col] = pd.to_numeric(df_unificado[col], errors='coerce')

    return df_unificado
# --- 4. LÓGICA PRINCIPAL ---
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    df = load_full_dataset()

    if not df.empty:
        st.title("🌊 Análisis de Hidrología CDMX")
        
        # Dashboard de KPIs (Esencia Corporativa)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Registros Totales", f"{len(df):,}")
        with c2: st.metric("Periodo", "2025 - 2026")
        with c3: st.metric("Sensores Únicos", df['id_sensor_tuya'].nunique() if 'id_sensor_tuya' in df.columns else "N/A")
        with c4: st.metric("Estado Sistema", "Online", delta="Óptimo")

        # --- CONFIGURACIÓN DEL AGENTE CON GEMINI 2.5 ---
        # Usamos gemini-2.5-flash para balancear velocidad y precisión en tablas grandes
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            max_retries=2
        )

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            prefix="""Eres el analista senior de Hidrología de Rotoplas. 
            Tienes acceso a datos de 2025 y 2026 unificados. 
            Cuando grafiques, usa el color azul corporativo (#004C97).
            Siempre usa plt.show() al final de tus scripts de Python."""
        )

        # INTERFAZ DE CHAT
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ej: Compara el consumo de Iztapalapa entre mayo 2025 y mayo 2026"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Procesando histórico multiaño..."):
                    try:
                        # Ejecutar agente
                        response = agent.run(prompt)
                        st.markdown(response)
                        
                        # Capturar gráfico si existe
                        fig = plt.gcf()
                        if fig.get_axes():
                            st.pyplot(fig)
                        plt.close('all')
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error en el modelo: {e}")
    else:
        st.error("No se pudieron cargar los archivos Parquet. Verifica que estén en la misma carpeta que este script.")
else:
    st.info("💡 Por favor, ingresa tu API Key en el panel izquierdo para activar el análisis histórico.")
