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

    if prompt := st.chat_input("Ej: ¿Hay fugas en Iztapalapa y cuánto durará el agua?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("1. El Analista está extrayendo datos..."):
                try:
                    # Usamos invoke y extraemos solo el texto de salida
                    result = agent_analista.invoke({"input": f"Reporte de estado actual para: {prompt}"})
                    res_analista = result["output"] 
                except Exception as e:
                    # Si falla el parsing, intentamos recuperar lo que haya pensado
                    res_analista = "No se pudo procesar el análisis detallado, pero el sistema está revisando los datos."
                    print(f"Error en analista: {e}")

            with st.spinner("2. El ML Engineer está analizando riesgos..."):
                # Reducimos la complejidad del prompt para evitar el ValueError
                query_ml = f"Basado en estos datos: {res_analista}. Responde la duda del usuario: {prompt}"
                res_ml = agent_ml.run(query_ml)
    
            with st.spinner("Consultando al equipo de expertos..."):
                
                # EJECUCIÓN EN CASCADA (Orquestación simple)
                # Paso 1: El analista extrae los hechos actuales
                res_analista = agent_analista.run(f"Reporte de estado actual para: {prompt}")
                
                # Paso 2: El ML Engineer busca anomalías o tendencias basadas en la pregunta
                res_ml = agent_ml.run(f"Analiza tendencias de riesgo o predicciones para: {prompt}. Datos actuales: {res_analista}")
                
                # Paso 3: El Architect consolida todo
                prompt_final = f"""
                Usuario pregunta: {prompt}
                Datos del Analista: {res_analista}
                Análisis del ML Engineer: {res_ml}
                Genera una respuesta ejecutiva y clara.
                """
                respuesta_final = llm.predict(prompt_final)
                
                st.markdown(respuesta_final)

                # Manejo de gráficos (Si el analista generó uno)
                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig)
                    plt.close(fig)

            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
else:
    st.sidebar.info("Ingresa tu API Key para activar el panel multi-agente.")
