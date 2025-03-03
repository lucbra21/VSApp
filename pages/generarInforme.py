import streamlit as st
import common.login as login
import common.generatePdf as pdf

login.generarLogin()
if 'usuario' in st.session_state:
    st.header('Generador de Smart Report')
    
    # Agregar un input para el link con key único
    link = st.text_input("Ingrese el link del perfil del jugador de https://fbref.com/:", 
                        placeholder="https://fbref.com/en/players/idplayer/name-player",
                        key="input_link_pagina1")
    
    # Agregar botón para ver el informe con key único
    if st.button("Generar Informe", key="btn_ver_informe_pagina1"):
        # Generar y descargar el PDF
        if link:  # Verifica que se haya ingresado un link
            state,filename=pdf.generar_pdf(link)
            if state:
                # Abrir y leer el archivo para streamlit
                with open(filename, "rb") as f:
                    bytes = f.read()
                    
                # Botón de descarga en Streamlit
                st.download_button(
                    label="Descargar PDF",
                    data=bytes,
                    file_name='smart_report.pdf',
                    mime='application/octet-stream',
                    key="download_pdf"
                )
                st.success("PDF generado exitosamente!")
                st.markdown(f"[Abrir Link Original]({link})", unsafe_allow_html=True)
        else:
            st.error("Por favor, ingrese un link válido.")