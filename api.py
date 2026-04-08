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
st.set_page_config(page_title="Análisis Hidrología | Rotoplas", layout="wide")
st.title("🌊 Análisis de Hidrología CDMX")
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background-color: #F4F7F9;
    }
    
    /* Personalización del Sidebar */
    [data-testid="stSidebar"] {
        background-color: #004C97;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Títulos y fuentes */
    h1 {
        color: #004C97;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    
    /* Estilo de métricas (KPIs) */
    div[data-testid="stMetricValue"] {
        color: #00A9E0;
    }
    
    /* Botones estilo Rotoplas */
    .stButton>button {
        background-color: #00A9E0;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
    }
    </style>
    """, unsafe_allow_html=True)
with st.sidebar:
    st.image('Logo_de_Rotoplas.svg.png', width=200)
    st.markdown("---")
    st.subheader("🔑 Configuración")
    api_key = st.text_input("Introduce tu Google API Key", type="password")

    st.markdown("---")
    st.subheader("📍 Filtros de Análisis")
    alcaldia = st.selectbox("Seleccionar Alcaldía", ["Iztapalapa", "Benito Juárez", "Coyoacán"])
    periodo = st.date_input("Rango de fechas")
# 1. Configuración de API Key (vía Secrets o Sidebar)
# Añadimos un 'key' único al final
#api_key = st.sidebar.text_input("Introduce tu Google API Key", type="password", key="api_key_corporativa")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # 2. Carga de datos

    @st.cache_data
    def load_data():
        #file_id = '1bq9eLLaT5a386AhtGEKz2QuxzGzGwuAD'
        #url = f'https://drive.google.com/uc?id={file_id}'
        #return pd.read_csv(url)
        return pd.read_parquet("Base_CDMX_ultra.parquet")

    df = load_data()
    st.write("### Vista previa de los datos", df.head(5))

    # 3. Inicialización del Agente
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=None,
        timeout=None,
        max_retries=2,)
        custom_prefix = 
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        prefix=Prefix_ag,
        allow_dangerous_code=True,
        handle_parsing_errors=True
    )


# 4. Interfaz de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Si el mensaje guardado tenía un gráfico, aquí podrías manejarlo, 
            # pero por ahora enfoquémonos en la respuesta actual.

    if prompt := st.chat_input("¿Qué quieres saber sobre el consumo hídrico?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 1. Ejecutamos el agente UNA SOLA VEZ
            # Usamos un st.spinner para que el usuario sepa que está pensando
            with st.spinner("Analizando datos..."):
                response = agent.run(prompt)
                st.markdown(response)

            # 2. TRUCO DE GRÁFICOS: Verificamos el buffer de Matplotlib
            fig = plt.gcf() 
            if fig.get_axes(): # Si el agente dibujó algo...
                st.pyplot(fig)
                plt.close(fig) # Cerramos la figura para liberar memoria
            else:
                plt.close(fig) # Cerramos aunque esté vacía para evitar basura visual

            # 3. Guardar la respuesta en el historial
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.sidebar.info("Ingresa tu Google API Key para activar el agente.")
    st.warning("Esperando API Key...")
