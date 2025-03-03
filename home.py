import streamlit as st
import common.login as login

st.header('Página principal')
login.generarLogin()
if 'usuario' in st.session_state:
    st.subheader('Información página principal')
    st.write('Generador de reportes con inteligencia artificial generativa')