from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import gradio as gr
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
import asyncio

# Cargar las variables de entorno
load_dotenv(override=True)

# Definicion del estado de grafo
class Estado(TypedDict):
    messages: Annotated[list, add_messages] #Historial de mensajes 

# Configuracion de Pushover para notificaciones 
toke_pushover = os.getenv("PUSHOVER_TOKEN")
user_pushover = os.getenv("PUSHOVER_USER")
url_pushover = "https://api.pushover.net/1/messages.json"

@tool
def send_pushover(message: str) -> str:
    "Util cuando quieres enviar una notificacion push al usuario"
    try:
        requests.post(url_pushover, data={
            "token": toke_pushover,
            "user": user_pushover,
            "message": message
        })
    except Exception as e:
        print(f"Error al enviar notificacion push: {e}")



# Variables globales para el navegador
playwright_instance = None
browser_instance = None

async def inicializar_navegador():
    """Inicializa el navegador Playwright"""
    global playwright_instance, browser_instance
    from playwright.async_api import async_playwright
    
    playwright_instance = await async_playwright().start()
    browser_instance = await playwright_instance.chromium.launch(headless=False)
    return browser_instance

async def cerrar_navegador():
    """Cierra el navegador y Playwright"""
    global playwright_instance, browser_instance
    if browser_instance:
        await browser_instance.close()
    if playwright_instance:
        await playwright_instance.stop()

async def main_async():
    """Función principal asíncrona para ejecutar la aplicación"""
    print("Inicializando agente con navegador Playwright...")
    
    try:
        # Inicializar navegador
        browser = await inicializar_navegador()
        
        # Obtener herramientas del navegador 
        conjunto_herramientas = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        herramientas_navegador = conjunto_herramientas.get_tools()

        # Combinar todas las herramientas
        all_herramientas = herramientas_navegador + [send_pushover]

        # Configurar LLM con Groq (gratuito)
        llm = ChatOpenAI(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        llm_con_herramientas = llm.bind_tools(all_herramientas) 

        def chatbot(estado: Estado):
            return {"messages": [llm_con_herramientas.invoke(estado["messages"])]}
        
        # Construir el grafo 
        constructor_grafo = StateGraph(Estado)
        constructor_grafo.add_node("chatbot", chatbot)
        constructor_grafo.add_node("tools", ToolNode(tools=all_herramientas))
        constructor_grafo.add_conditional_edges("chatbot", tools_condition)
        constructor_grafo.add_edge("tools", "chatbot")
        constructor_grafo.add_edge(START, "chatbot")

        # Compilar con memoria 
        memoria = MemorySaver()
        grafo = constructor_grafo.compile(checkpointer=memoria)
        
        print("Agente inicializado correctamente!")

        # Crear interfaz de Gradio con función async
        async def chat_wrapper_async(entrada_usuario: str, historial):
            configuracion = {"configurable": {"thread_id": "10"}}
            resultado = await grafo.ainvoke(
                {"messages": [{"role": "user", "content": entrada_usuario}]},
                config=configuracion
            )
            return resultado["messages"][-1].content

        # Lanzar interfaz
        print("Iniciando interfaz de Gradio...")
        demo = gr.ChatInterface(
            chat_wrapper_async,
            title="Agente Web con Playwright",
            description="Agente con capacidades de navegación web y notificaciones push"
        )

        # Lanzar Gradio sin bloquear
        demo.launch(prevent_thread_lock=True)

        # Mantener el loop corriendo
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nCerrando aplicación...")
    finally:
        await cerrar_navegador()
        print("Navegador cerrado.")

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nAplicación terminada.")

    
