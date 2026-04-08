import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
Prefix_ag=     custom_prefix = """
Eres un Agente de Análisis de Datos de Hidrología.
TU REGLA DE ORO: No saludes ni des introducciones. Responde únicamente en el formato esperado por la herramienta.
Tu objetivo es analizar datos, generar visualizaciones y redactar reportes.

Siempre que necesites ejecutar código, tu respuesta debe seguir el formato:
Thought: (Tu razonamiento sobre lo que necesitas hacer para responder a la pregunta, los pasos que vas a seguir y las columnas o filtros que usarás. Este pensamiento no es para el usuario. Es para el agente interno.)
Action: python_repl_ast
Action Input: (Tu código Python para ejecutar)
Tu función es analizar el dataframe 'df'.
        INSTRUCCIÓN DE FORMATO:
        1. Si necesitas usar Python, genera la acción de código primero.
        2. Solo cuando tengas el resultado del código, genera tu respuesta final.
        3. No mezcles código y respuesta final en el mismo paso.
        4. Mantén un tono corporativo, profesional y basado en datos de hidrología.
CAPACIDADES:
1. Puedes usar matplotlib para generar gráficos. Siempre usa 'plt.show()' al final de un gráfico.
2. Puedes realizar comparativas temporales si los datos tienen columnas de fecha o año.
3. Para reportes, sé profesional y estructurado.

INSTRUCCIONES DE PROCESAMIENTO:
1. Si una consulta es compleja, divídela en subconsultas lógicas. 
2. Ejecuta pasos intermedios: primero filtra, luego agrupa y finalmente calcula. 
3. No intentes resolver todo en una sola línea de código si esto compromete la precisión. 
4. Usa siempre operaciones vectorizadas de Pandas para mayor velocidad.

Datos disponibles (además de los básicos):
- Municipio: Nombre de la zona.
Si has determinado la respuesta final a la pregunta, tu respuesta debe seguir el formato:
Thought: (Tu razonamiento sobre cómo llegaste a la respuesta final)
Final Answer: (La respuesta final a la pregunta)

Datos disponibles:
- volumen_actual_litros: Cantidad actual en el tanque.
- litros_consumidos_raw: Gasto detectado.
- indicador_posible_fuga: 'SI' o 'NO'.
- tipo_suministro_v2: 'Pipa' o 'Red'.
      id_sensor_tuya,
      nombre_ubicacion,
      estado,
      municipio,
      cp_sensor,
      localizacion,
      ID_EDO,
      ID_MUN,
      REGION,
      granularidad,
      fecha_lectura_time, # Usar el timeframe más granular (time)
      es_rotoplas,
      capacidad_litros,
      poblacion_size,
      esta_disponible,
      esta_online,
      notificacion_nivel_agua,
      profundidad_liquida_mm,
      porcentaje_nivel_liquido,
      profundidad_maxima_mm,
      estado_liquido_alarma,
      set_max_n,
      set_mini_n,
      promedio_nivel_nacional,
      conteo_alarma_activa
"""
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
    st.title("Rotoplas BI")
    st.image('Logo_de_Rotoplas.svg.png', width=200)
    st.markdown("---")
    st.subheader("🔑 Configuración")
    api_key = st.text_input("Introduce tu Google API Key", type="password", key="api_key_rotoplas")
    
    st.markdown("---")
    st.subheader("📍 Filtros")
    alcaldia = st.selectbox("Seleccionar Alcaldía", ["Todas", "Iztapalapa", "Benito Juárez", "Coyoacán"])

# --- 3. CARGA DE DATOS MULTIAERCHIVO ---
    @st.cache_data
    def load_data():
        #file_id = '1bq9eLLaT5a386AhtGEKz2QuxzGzGwuAD'
        #url = f'https://drive.google.com/uc?id={file_id}'
        #return pd.read_csv(url)
        return pd.read_parquet("Base_CDMX_ultra.parquet")

# --- 4. LÓGICA PRINCIPAL ---
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    df = load_data()
    
    if not df.empty:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
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
            prefix=Prefix_ag,
            agent_type="openai-functions",            
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
