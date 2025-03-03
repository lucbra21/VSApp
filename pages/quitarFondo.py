import streamlit as st
import common.login as login
from common.removeBackground import removeBackground
import requests
from PIL import Image
from io import BytesIO
import os

def download_image(url):
    """Descarga una imagen desde una URL"""
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error al descargar la imagen: {str(e)}")
        return None

login.generarLogin()
if 'usuario' in st.session_state:
    st.header('Generador de Smart Report')
    
    # Agregar un input para el link
    link = st.text_input(
        "Ingrese el link del perfil del jugador de https://fbref.com/:", 
        placeholder="https://fbref.com/en/players/idplayer/name-player",
        key="input_link_pagina1"
    )
    
    # Agregar campo para subir imagen
    uploaded_file = st.file_uploader("O suba una imagen directamente:", type=['png', 'jpg', 'jpeg'])
    
    # Botón para quitar fondo
    if st.button("Quitar Fondo", key="btn_quitar_fondo"):
        try:
            if uploaded_file:
                # Si se subió un archivo directamente
                img_path = removeBackground(uploaded_file)
                if img_path and os.path.exists(img_path):
                    st.success("Fondo eliminado exitosamente!")
                    st.image(img_path)
                else:
                    st.error("Error al procesar la imagen")
            elif link:
                # Verificar que el link sea válido
                if not link.startswith("https://fbref.com/"):
                    st.error("Por favor, ingrese un link válido de fbref.com")
                else:
                    # Procesar imagen desde el link
                    img_path = removeBackground(link)
                    if img_path and os.path.exists(img_path):
                        st.success("Fondo eliminado exitosamente!")
                        st.image(img_path)
                        st.markdown(f"[Abrir Link Original]({link})", unsafe_allow_html=True)
                    else:
                        st.error("Error al procesar la imagen del link")
            else:
                st.error("Por favor, suba una imagen o ingrese un link válido")
                
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")