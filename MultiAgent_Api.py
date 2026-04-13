import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
#import vertexai # Si migras a Vertex, si no mantén langchain_google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. DEFINICIÓN DE LÓGICAS (PROMPTS ESPECIALIZADOS) ---

PREFIX_ANALISTA = """Eres un Data Analyst experto en Hidrología. 
Tu única fuente de verdad es el dataframe 'df'. 
TU FUNCIÓN: Calcular promedios, filtrar alcaldías y generar gráficos.
REGLA: Si te piden predecir el futuro o explicar por qué ocurre algo, pasa el dato técnico pero no especules.
1. Puedes usar matplotlib para generar gráficos. Siempre usa 'plt.show()' al final de un gráfico.
IMPORTANTE: Siempre termina tu respuesta con el formato 'Final Answer: [tu respuesta aquí]'. No añadas texto después de la respuesta final."""

PREFIX_ML_ENGINEER = """Eres un Machine Learning Engineer. 
TU FUNCIÓN: Analizar tendencias y detectar anomalías.
LÓGICA: 
1. Si 'indicador_posible_fuga' es 'SI', calcula cuánto volumen se pierde por hora.
2. Si piden predicciones, usa el promedio de 'litros_consumidos_raw' para proyectar el vaciado del tanque.
3. Identifica patrones de riesgo en los datos."""

PREFIX_ARCHITECT = """Eres el LLM Architect (Director de Orquesta). 
TU FUNCIÓN: Recibir los hallazgos del Analista y del ML Engineer para dar una respuesta coherente al usuario.
REGLA: Traduce los términos técnicos a recomendaciones prácticas para Rotoplas.
IMPORTANTE: Siempre termina tu respuesta con el formato 'Final Answer: [tu respuesta aquí]'. No añadas texto después de la respuesta final."""

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Análisis Hidrología | Multi-Agente", layout="wide")
st.title("🌊 Sistema de Inteligencia Hídrica CDMX")

# Sidebar y Estilos (Se mantienen igual a tu código original)
with st.sidebar:
    st.image('Logo_de_Rotoplas.svg.png', width=200)
    api_key = st.text_input("Introduce tu Google API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    @st.cache_data
    def load_data():
        return pd.read_parquet("Base_CDMX_ultra.parquet")

    df = load_data()

    # --- 3. INICIALIZACIÓN DE LOS 3 AGENTES ---
    llm = ChatGoogleGenerativeAI( model="gemini-2.5-flash", temperature=0,
        max_output_tokens=None,
        timeout=None,
        max_retries=2)

    # Agente 1: El Analista (Basado en tu código de Pandas Agent)
    agent_analista = create_pandas_dataframe_agent(
        llm=llm, 
        df=df, 
        verbose=True, 
        prefix=PREFIX_ANALISTA,
        agent_type="tool-calling",
        allow_dangerous_code=True, 
        handle_parsing_errors=True,
        max_iterations=10
    )

    # Agente 2: El ML (También basado en Pandas para cálculos predictivos)
    agent_ml = create_pandas_dataframe_agent(
        llm=llm, 
        df=df, 
        verbose=True, 
        prefix=PREFIX_ML_ENGINEER,
        agent_type="tool-calling",
        allow_dangerous_code=True, 
        handle_parsing_errors=True,
        max_iterations=10
    )

    # --- 4. INTERFAZ DE CHAT Y LÓGICA DE ORQUESTACIÓN ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ... (Todo tu código anterior de carga de datos y definición de agentes) ...

if prompt := st.chat_input("¿Qué quieres saber sobre el consumo hídrico?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando al equipo de expertos..."):
            
            # --- PASO 1: Ejecuta el Analista ---
            res_analista_raw = agent_analista.invoke({"input": prompt})
            res_analista = res_analista_raw["output"]
            
            # --- PASO 2: Ejecuta el ML Engineer ---
            res_ml_raw = agent_ml.invoke({"input": f"Analiza riesgos para: {prompt}. Datos: {res_analista}"})
            res_ml = res_ml_raw["output"]

            # --- PASO 3: AQUÍ COLOCAS EL CÓDIGO NUEVO (CONSOLIDACIÓN) ---
            # Aseguramos que los resultados sean strings y no objetos de LangChain
            contexto_limpio = f"""
            Datos del Analista: {str(res_analista)}
            Análisis del ML Engineer: {str(res_ml)}
            """

            prompt_final = f"""
            Usuario pregunta: {prompt}
            Contexto técnico: {contexto_limpio}
            Genera una respuesta ejecutiva y clara.
            """

            # Ejecución corregida
            respuesta_final = llm.invoke(prompt_final).content
            # ---------------------------------------------------------

            # --- PASO 4: Mostrar al usuario ---
            st.markdown(respuesta_final)

            # (Opcional) Manejo de gráficos si el analista creó uno
            fig = plt.gcf()
            if fig.get_axes():
                st.pyplot(fig)
                plt.close(fig)

        st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
else:
    st.sidebar.info("Ingresa tu API Key para activar el panel multi-agente.")
