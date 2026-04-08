import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
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

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🚰 Rotoplas BI")
    st.markdown("---")
    st.subheader("🔑 Configuración")
    api_key = st.text_input("Introduce tu Google API Key", type="password", key="api_key_rotoplas")
    
    st.markdown("---")
    st.subheader("📍 Filtros")
    alcaldia = st.selectbox("Seleccionar Alcaldía", ["Todas", "Iztapalapa", "Benito Juárez", "Coyoacán"])

# --- 3. CARGA DE DATOS MULTIAERCHIVO ---
@st.cache_data
def load_full_data():
    archivos = [
        "2025Q1.parquet", "2025Q2P1.parquet", "2025Q2P2.parquet",
        "2025Q3P1.parquet", "2025Q3P2.parquet", "2026Q1.parquet"
    ]
    
    list_dfs = []
    for f in archivos:
        if os.path.exists(f):
            list_dfs.append(pd.read_parquet(f))
    
    if not list_dfs:
        return pd.DataFrame()
    
    df_final = pd.concat(list_dfs, ignore_index=True)
    
    # Normalización de fechas para el agente
    if 'fecha_lectura_time' in df_final.columns:
        df_final['fecha_lectura_time'] = pd.to_datetime(df_final['fecha_lectura_time'], errors='coerce')
        df_final = df_final.dropna(subset=['fecha_lectura_time']).sort_values('fecha_lectura_time')
    
    return df_final

# --- 4. LÓGICA PRINCIPAL ---
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    df = load_full_data()

    if not df.empty:
        st.title("🌊 Análisis de Hidrología CDMX")
        
        # KPIs rápidos
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Registros Cargados", f"{len(df):,}")
        with c2: st.metric("Rango Temporal", "2025 - 2026")
        with c3: st.metric("Estatus", "Conectado")

        # Inicialización del Agente optimizado
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0
        )

        # Usamos AgentType.ZERO_SHOT_REACT_DESCRIPTION para mayor estabilidad en el parseo
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        )

        # Interfaz de Chat
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("¿Qué quieres saber sobre el consumo hídrico?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analizando base histórica..."):
                    try:
                        # Ejecución del agente
                        response = agent.invoke(prompt)
                        # Dependiendo de la versión de LangChain, el resultado está en 'output'
                        res_text = response.get("output", str(response))
                        st.markdown(res_text)
                        
                        # Manejo de gráficos
                        fig = plt.gcf()
                        if fig.get_axes():
                            st.pyplot(fig)
                        plt.close('all')
                        
                        st.session_state.messages.append({"role": "assistant", "content": res_text})
                    except Exception as e:
                        st.error(f"Hubo un problema al procesar la respuesta: {e}")

    else:
        st.error("No se encontraron los archivos .parquet. Asegúrate de que estén en la misma carpeta.")
else:
    st.warning("Introduce tu API Key para comenzar.")
