import streamlit as st
import pandas as pd

def generarMenu(usuario):
    """Genera el menú dependiendo del usuario

    Args:
        usuario (str): usuario utilizado para generar el menú
    """
    with st.sidebar:
        usuarios = pd.read_csv('data/usuarios.csv')
        
        # filtrar en usuarios por usuarios = usuario
        result = usuarios[usuarios['usuario'] == usuario]

        # Verificar si se encontró el usuario
        if not result.empty:  # Aquí está el cambio principal
            # Asumiendo que 'nombre' es una columna en tu CSV
            nombre = result['nombre'].iloc[0]  # Y aquí la forma correcta de acceder al valor
            # Mostrar el nombre del usuario
            st.write(f"Hola **:blue-background[{nombre}]** ")
        else:
            st.error("Usuario no encontrado en la base de datos.")

        # Mostrar los enlaces de páginas
        st.page_link("home.py", label="Home", icon=":material/home:")
        st.subheader("Opciones")
        st.page_link("pages/generarInforme.py", label="Report", icon=":material/sell:")
        st.page_link("pages/quitarFondo.py", label="Remove Background", icon=":material/sell:")

        # Botón para cerrar la sesión
        btnSalir = st.button("Salir", key="btn_salir_menu")
        if btnSalir:
            st.session_state.clear()
            st.rerun()