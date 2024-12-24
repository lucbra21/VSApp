#LIBRERIAS NECESARIAS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import tabulate
import requests
from bs4 import BeautifulSoup
import re
import ast
import math
from urllib.parse import urljoin
from datetime import datetime
import time
from urllib.request import urlopen
from fpdf import FPDF
from common.removeBackground import removeBackground
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from PIL import Image
from io import BytesIO
import os
import tempfile
import ollama
from mplsoccer import PyPizza, add_image, FontManager, Radar, grid
import LanusStats as ls
fbref = ls.Fbref()


#FUENTES PARA LOS PERCENTILES
font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/'
                          'src/hinted/Roboto-Regular.ttf')
font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/'
                          'src/hinted/Roboto-Italic.ttf')
font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                        'RobotoSlab[wght].ttf')

#FUENTES PARA LOS RADARES COMPARTIVOS
URL1 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-Regular.ttf')
serif_regular = FontManager(URL1)
URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-ExtraLight.ttf')
serif_extra_light = FontManager(URL2)
URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
        'RubikMonoOne-Regular.ttf')
rubik_regular = FontManager(URL3)
URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
robotto_thin = FontManager(URL4)
URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
        'RobotoSlab%5Bwght%5D.ttf')
robotto_bold = FontManager(URL5)

class PDF(FPDF):
    def __init__(self, player_name):
        super().__init__()
        self.player_name = player_name
        
    def header(self):
        self.set_fill_color(50, 50, 50)  # Fondo negro
        self.rect(0, 0, self.w, self.h, 'F')  # Crear fondo negro

        # Cargar el archivo original
        ruta_origen = "assets/logo_convertido-removebg-preview.png"
        ruta_destino = "assets/logo_convertido.png"

        # Convertir a un PNG estándar
        try:
            img = Image.open(ruta_origen)
            img.convert("RGBA").save(ruta_destino, format="PNG")
            print(f"Archivo convertido y guardado en {ruta_destino}")
        except Exception as e:
            print(f"Error al convertir el archivo: {e}")
        self.image("assets/logo_convertido.png", x=160, y=8, w=33)

        # Configurar fuente y color para el título
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)  # Color del texto en blanco
        # Añadir el título
        title = f'Informe de {self.player_name}'
        title_width = self.get_string_width(title)  # Obtener el ancho del texto
        cell_margin = 3  # Margen extra alrededor del texto
        cell_width = title_width + cell_margin
        self.cell(cell_width, 12, title, border=1, align="L")
        self.ln(15)  # Añadir un salto de línea
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(255, 255, 255)  # Color del texto en blanco
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        # Título centrado
        self.set_font("Arial", "B", 14)
        self.set_text_color(255, 255, 255)  # Color blanco para el título
        title_width = self.get_string_width(title)
        self.set_x((self.w - title_width) / 2 + 2)  # Mueve 5 unidades a la derecha
        self.cell(title_width, 10, title, 0, 1, 'C')
        self.ln(5)  # Salto de línea adicional después del título

    def chapter_title2(self, title):
        # Título centrado sin el ajuste de margen
        self.set_font("Arial", "B", 14)
        self.set_text_color(255, 255, 255)  # Color blanco para el título
        title_width = self.get_string_width(title)
        self.cell(title_width, 10, title, 0, 1, 'C')
        self.ln(5)  # Salto de línea adicional después del título


def extract_player_info(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extraer información básica
        player_info = {}
        
        # Intentamos primero extraer el nombre del jugador de <h1><span>
        player_name = soup.find('h1').find('span')
        if player_name:
            player_info['Player'] = player_name.text.strip()
        else:
            # Si no encontramos el nombre en <h1><span>, buscamos el primer <strong> dentro de <p>
            strong_name = soup.find('p').find('strong')
            if strong_name:
                player_info['Player'] = strong_name.text.strip()
            else:
                player_info['Player'] = 'Nombre no encontrado'  # Valor por defecto si no se encuentra el nombre

        # Extraer la posición del jugador
        position_element = soup.select_one('p:-soup-contains("Position")')
        if position_element:
            player_info['Position'] = position_element.text.split(':')[1].split(' ')[0].strip()
        else:
            player_info['Position'] = 'Posición no encontrada'

    # Extraer el equipo del jugador
        paragraphs = soup.find_all('p')
        player_team = None
        # Recorremos todos los párrafos
        for p in paragraphs:
        # Verificamos si el párrafo contiene el texto 'Club :'
            if 'Club:' in p.text:
            # Encontramos el enlace dentro del párrafo
                player_team = p.find('a').text.strip()
                break
        player_info['Team'] = player_team if player_team else 'Equipo no encontrado'

        # Extraer la fecha de nacimiento y calcular la edad
        birthday = soup.select_one('span[id="necro-birth"]')
        if birthday:
            birthday = birthday.text.strip()
            player_info['Age'] = (datetime.now() - datetime.strptime(birthday, '%B %d, %Y')).days // 365
        else:
            player_info['Age'] = 'Edad no encontrada'

        # Extraer la URL de la imagen del jugador
        media_item_div = soup.find('div', class_='media-item')  # Verificar si el div existe
        if media_item_div:  # Si el div existe, buscar el tag <img>
            img_tag = media_item_div.find('img')
            if img_tag and 'src' in img_tag.attrs:  # Verificar si el <img> existe y tiene el atributo 'src'
                player_info['Photo_URL'] = img_tag['src']
            else:
                player_info['Photo_URL'] = ""  # Si no hay <img> o no tiene 'src'
        else:
            player_info['Photo_URL'] = ""  # Si no hay div con la clase 'media-item'
        


        def extract_correct_scouting_report_url(url):
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    argentina_league_link = soup.find('a', href=True, text='2024 Liga Profesional Argentina')

                    if argentina_league_link:
                        relative_url = argentina_league_link['href']
                        full_url = urljoin(url, relative_url)
                        return full_url

                return None
        # Extraer la URL del scouting report
        scouting_report_url = extract_correct_scouting_report_url(url)
        player_info['Report_URL_liga_argentina'] = scouting_report_url

        # EXTRAER LOS MINUTOS JUGADOS DESDE EL SCOUTING REPORT
        if scouting_report_url:
            try:
                # Realizar la solicitud para obtener el contenido del scouting report
                report_response = requests.get(scouting_report_url)
                if report_response.status_code == 200:
                    report_soup = BeautifulSoup(report_response.text, 'html.parser')

                    # Buscar el div con el estilo específico
                    minutos = report_soup.find('div', style="max-width:500px")
                    if minutos:
                        strong_tag = minutos.find('strong')
                        if strong_tag:
                            player_info['Minutos_jugados_reporte'] = strong_tag.text.strip()
                        else:
                            player_info['Minutos_jugados_reporte'] = ""  # Si no encuentra <strong>, asigna vacío
                    else:
                        player_info['Minutos_jugados_reporte'] = ""  # Si no encuentra el div, asigna vacío
                else:
                    player_info['Minutos_jugados_reporte'] = "No se pudo acceder al reporte"
            except Exception as e:
                print(f"Error al extraer los minutos jugados desde el reporte: {e}")
                player_info['Minutos_jugados_reporte'] = "Error al extraer minutos"

        return player_info

def obtener_datos_90(url):
        datos_jugador=fbref.get_player_percentiles(url)
        return datos_jugador

def limpiar_nivel_df(df):
        if isinstance(df.columns, pd.MultiIndex):
            return df.droplevel(axis=1, level=0)
        return df

def procesar_tabla(df):
            gk_stats = ['Touches','Goals Against', 'Save Percentage',  'PSxG/SoT', 'PSxG-GA', 'Crosses Stopped %']
            other_stats = ['Non-Penalty Goals', 'Shots Total','Assists','Shot-Creating Actions',
                        'Pass Completion %','Progressive Passes', 'Progressive Carries','Touches (Att Pen)', 
                        'Tackles', 'Interceptions','Blocks', 'Aerials Won']
            todas_las_stats = gk_stats + other_stats

            mapeo_nombres = {
                'Touches': "Toques",
                'Goals Against':'Goles en contra',
                'Save Percentage': '% Paradas',
                'PSxG/SoT':'PSxG/SoT',
                'PSxG-GA': 'PSxG-GA',        
                'Crosses Stopped %':'% Cruces detenidos',
                
                'Non-Penalty Goals': 'Goles sin penalizacion',
                'Shots Total': "Total tiros",
                'Assists': "Asistencias",
                'Shot-Creating Actions': "Acciones para la creación de tiros",
                'Pass Completion %': '% Pases completados',
                'Progressive Passes': 'Pases progresivos',
                'Progressive Carries':'Acarreos progresivos',
                'Touches (Att Pen)': "Toques (Ataq. pen.)",
                'Tackles': "Derribos",
                'Interceptions': "Intercepciones",
                "Blocks": "Bloqueos",
                "Aerials Won": "Duelos áreos ganados"
            }

            tabla_filtrada = df[df['Statistic'].isin(todas_las_stats)].copy()
            tabla_filtrada['Statistic'] = tabla_filtrada['Statistic'].map(mapeo_nombres).fillna(tabla_filtrada['Statistic'])

                    
            # Limpiar y convertir la columna 'Per 90' (si hay valores con '%')
            tabla_filtrada['Per 90'] = tabla_filtrada['Per 90'].replace('%', '', regex=True)  # Eliminar el símbolo '%'
            tabla_filtrada['Per 90'] = pd.to_numeric(tabla_filtrada['Per 90'], errors='coerce')  # Convertir a numérico, NaN para valores no convertibles


            return tabla_filtrada



def obtener_indice_tabla_sp(url):
        """
        Función para determinar automáticamente el índice de la tabla a extraer según las opciones de filtrado.
        Si hay filtros, se devuelve el índice 4, si no, se devuelve el índice 2.
        """
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Buscar los filtros disponibles en la página (asumimos que los filtros tienen la clase '.sr_preset')
            filter_buttons = soup.select(".filter.switcher a.sr_preset")
            # Mostrar cuántos filtros se han encontrado para depuración
            print(f"Filtros encontrados: {len(filter_buttons)}")
            
            # Si hay más de un filtro, usamos el índice 4
            if len(filter_buttons) > 3:
                print("Hay más de 3 filtro, se usará el índice 4.")
                return 2  # Seleccionamos la tabla con índice 4
            else:
                print("Hay solo 3 filtros, se usará el índice 2.")
                return 1  # Seleccionamos la tabla con índice 2
        else:
            print(f"Error al cargar la página: {response.status_code}")
            return None

def extraer_similar_players(url):
        """
        Función para extraer la tabla correspondiente usando el índice detectado automáticamente.
        """
        # Determinar el índice de la tabla automáticamente
        table_index = obtener_indice_tabla_sp(url)
        
        if table_index is not None:
            # Extraer la tabla según el índice automáticamente
            tables = pd.read_html(url)
            
            # Verificamos que la tabla seleccionada existe
            if len(tables) > table_index:
                return tables[table_index]
            else:
                print("No se encontró la tabla en el índice seleccionado.")
                return None
        else:
            print("No se pudo determinar el índice de la tabla.")
            return None


    # Función principal para completar la información del jugador
def get_player_with_stats(url):
        # Extraer información básica del jugador
        player_info = extract_player_info(url)

        # Extraer tablas desde el scouting report URL
        report_url = player_info.get('Report_URL_liga_argentina')
        if report_url:
            try:
                # Procesar tabla de métricas
                
                datos_jugador= obtener_datos_90(player_info['Report_URL_liga_argentina'])
                df_sin_nivel= limpiar_nivel_df(datos_jugador)
                df_procesado= procesar_tabla(df_sin_nivel)
                player_info['Tabla_metricas'] = df_procesado

                # Extraer métricas y percentiles
                player_info['Metricas'] = df_procesado['Statistic'].tolist()
                player_info['Per 90'] = list(map(float, df_procesado['Per 90'].tolist()))
            except Exception as e:
                print(f"Error al procesar la tabla de métricas desde {report_url}: {e}")

            try:
                # Procesar tabla de jugadores similares
                jugadores_similares = extraer_similar_players(report_url)
                player_info['Similar_players'] = jugadores_similares
            except Exception as e:
                print(f"Error al procesar la tabla de jugadores similares desde {report_url}: {e}")

        return player_info

def extraccion_equipos_fbref(liga='Primera Division Argentina', año= '2024'):
    df_teams= fbref.get_all_teams_season_stats('Primera Division Argentina', '2024', save_csv=False, stats_vs=False, change_columns_names=True, add_page_name=True)
    return df_teams

def filtrar_cambiar_nombres(df_teams):
    # Filtrar las columnas deseadas
    df= df_teams[['stats_Squad', 'stats_PlayingTime_90s',  'stats_PlayingTime_Min','stats_Performance_G-PK','shooting_Standard_Sh', 'stats_Performance_Ast','gca_SCA_SCA','passing_Total_Cmp%',
                    'passing_PrgP', 'possession_Carries_PrgC', 'possession_Touches_AttPen','defense_Tackles_Tkl','defense_Int', 'defense_Blocks_Blocks', 'misc_AerialDuels_Won', 
                    'keepersadv_Expected_PSxG+/-','keepersadv_Goals_GA','keepers_Performance_Save%' ,'keepersadv_Expected_PSxG/SoT','possession_Touches_Touches','keepersadv_Crosses_Stp%']]
    # Diccionario para renombrar columnas
    mapeo_columns= {
        'stats_Squad': 'Equipo', 'stats_PlayingTime_90s': '90s',  
        'stats_PlayingTime_Min':'Min','stats_Performance_G-PK': 'Goles sin penalizacion',
        'shooting_Standard_Sh':'Total tiros', 'stats_Performance_Ast': 'Asistencias',
        'gca_SCA_SCA': 'Acciones para la creación de tiros','passing_Total_Cmp%': '% Pases completados',
        'passing_PrgP': 'Pases progresivos', 'possession_Carries_PrgC': 'Acarreos progresivos',
        'possession_Touches_AttPen': 'Toques (Ataq. pen.)','defense_Tackles_Tkl': 'Derribos',
        'defense_Int': 'Intercepciones',  'defense_Blocks_Blocks': 'Bloqueos', 'misc_AerialDuels_Won': 'Duelos áreos ganados',
        'keepersadv_Expected_PSxG+/-':'PSxG-GA','keepersadv_Goals_GA':'Goles en contra','keepers_Performance_Save%': '% Paradas','keepersadv_Expected_PSxG/SoT': 'PSxG/SoT', 
        'possession_Touches_Touches': 'Toques', 'keepersadv_Crosses_Stp%': '% Cruces detenidos'
    }

    # Renombrar las columnas usando el método `rename`
    df_final = df.rename(columns=mapeo_columns)
    return df_final

def normalizar_columnas_90s (df_final):
    # Lista de columnas que NO quieres convertir
    columnas_excluidas = ['Equipo','Min', '% Pases completados', 'PSxG/SoT', '% Paradas', '% Cruces detenidos']

    # Iterar sobre todas las columnas excepto las excluidas
    for columna in df_final.columns:
        if columna not in columnas_excluidas and columna != '90s':  # Asegúrate de no modificar la columna '90s'
            # Convertir los valores de la columna a base de 90 minutos utilizando '90s'
            df_final[columna] = (df_final[columna] / df_final['90s']).round(2)
    return df_final

def dataframe_medias(df_final):
    df_copy= df_final.copy()
    # Eliminar la columna 'Equipo' antes de obtener la media
    df_sin_equipo = df_copy.drop(columns=['Equipo', '90s', 'Min'])

    # Calcular la media de todas las columnas
    media_columnas = df_sin_equipo.mean()

    # Convertir la serie de medias a un DataFrame
    df_media_columnas = media_columnas.to_frame().reset_index()

    # Renombrar las columnas para tener nombres más legibles
    df_media_columnas.columns = ['Columna', 'Media']

    # Mostrar el DataFrame con las medias
    return df_media_columnas.round(2)

def dividir_medidas_jugadores(df_media_columnas):
    # Seleccionar solo las primeras 11 columnas del DataFrame
       #df_valores_jugadores_campo = df_medias_liga.iloc[:12, :]
    
    df_valores_jugadores_delanteros = df_medias_liga.iloc[:4, :]
    df_valores_jugadores_centrocampistas = df_medias_liga.iloc[4:8, :]
    df_valores_jugadores_defensas = df_medias_liga.iloc[8:12, :]
    # Seleccionar solo las primeras 11 columnas del DataFrame
    df_valores_gk = df_medias_liga.iloc[12:, :]
    
    return df_valores_jugadores_delanteros, df_valores_jugadores_centrocampistas, df_valores_jugadores_defensas, df_valores_gk

df_teams= extraccion_equipos_fbref(liga='Primera Division Argentina', año= '2024')
df_final= filtrar_cambiar_nombres(df_teams)
df_final = normalizar_columnas_90s (df_final)
df_medias_liga= dataframe_medias(df_final)
df_valores_jugadores_delanteros, df_valores_jugadores_centrocampistas, df_valores_jugadores_defensas, df_valores_gk= dividir_medidas_jugadores(df_medias_liga)

especifico_a_general = {
    'GK': 'Porteros',

    'DF': 'Defensas',
    'DF (FB, right)':  'Defensas',
     'DF (FB, left)': 'Defensas',
     'DF-MF':  'Defensas',
     'DF-MF (FB, left)':  'Defensas',
     'DF-MF (FB, right)':  'Defensas',
     'DF-MF (CB-FB, right)': 'Defensas' ,
     'DF-MF (CB-FB, right)' :'Defensas' ,
     'DF-MF (CB-DM)':'Defensas',
     'DF (CB)':'Defensas',
     'DF (CB, left)':'Defensas' ,
     'DF (CB, right)':'Defensas' ,
     'DF (CB-FB)':'Defensas',
     'DF (CB-FB, right)': 'Defensas',
     'DF (CB-FB, left)':'Defensas',
     'DF-MF (FB-WM, right)':'Defensas',
    'DF-MF (FB-WM, left)':'Defensas',
    'DF-MF (DM)': 'Defensas',
    'DF-MF (AM-CM-WM)': 'Defensas',
    'DF-MF (CB)': 'Defensas',

    'MF': 'Centrocampistas',
    'MF (CM-WM)': 'Centrocampistas',
    'MF (CM-DM)': 'Centrocampistas',
    'DF-FW-MF (WM, right)':'Centrocampistas',
    'DF-FW-MF (WM, left)':'Centrocampistas',
    'MF (CM)':'Centrocampistas',
    'MF (AM)':'Centrocampistas',
    'MF (WM)':'Centrocampistas',
    'MF (CM, right)':'Centrocampistas',
    'MF (CM, left)':'Centrocampistas',
    'MF (WM, right)':'Centrocampistas',
    'MF (WM, left)':'Centrocampistas',
    'MF (DM)':'Centrocampistas',
    'MF (DM, left)':'Centrocampistas',
    'MF (AM-CM-WM)':'Centrocampistas',
    'MF (DM, right)':'Centrocampistas',
    'MF (CM-DM, right)':'Centrocampistas',
    'MF (CM-DM, left)':'Centrocampistas',
    'MF (AM-CM-DM-WM)':'Centrocampistas',

    'FW': 'Delanteros',
    'FW-MF': 'Delanteros',
    'DF-FW-MF (AM-WM)': 'Delanteros',
    'FW-MF (AM-WM)':'Delanteros',
    'FW-MF (AM)':'Delanteros',
    'FW-MF (AM, left)': 'Delanteros',
    'FW-MF (AM, right)':'Delanteros',
    'FW-MF (WM)':'Delanteros',
    'FW-MF (AM-CM-WM)':'Delanteros',
    'FW-MF (AM-WM)': 'Delanteros',
    'FW-MF (AM-WM, left)':'Delanteros',
    'FW-MF (AM-WM, right)':'Delanteros',
    'DF-FW-MF (CM-WM, left)': 'Delanteros'   

}

def obtener_valores_liga (player_info, especifico_a_general,df_valores_jugadores_delanteros, df_valores_jugadores_centrocampistas, df_valores_jugadores_defensas, df_valores_gk ):
    # Asumiendo que player_info['Position'] tiene el valor que indica la posición del jugador.
    posicion_especifica = player_info['Position']
    rol_general = especifico_a_general.get(posicion_especifica, 'Desconocido')  # Usamos el diccionario
    player_info['rol_general'] = rol_general

    # Paso 3: Seleccionar dinámicamente el DataFrame según el rol general
    if rol_general == 'Delanteros':
        df_valores = df_valores_jugadores_delanteros
    elif rol_general == 'Centrocampistas':
        df_valores = df_valores_jugadores_centrocampistas
    elif rol_general == 'Defensas':
        df_valores = df_valores_jugadores_defensas
    elif rol_general == 'Porteros':
        df_valores = df_valores_gk
    else:
        # Si no se encuentra un rol válido, asigna un DataFrame vacío
        df_valores = pd.DataFrame()

    # Paso 4: Obtener la columna 'Media' del DataFrame correspondiente
    if not df_valores.empty:
        values_liga = df_valores['Media'].to_list()
    else:
        values_liga = None  # O manejarlo según tu necesidad
    print(values_liga)
    return values_liga

# Parámetros específicos por rol
params_por_rol = {
        "Delanteros": ["Goles sin penalizacion", "Total tiros", "Asistencias", "Acciones para la creación de tiros"],
        "Centrocampistas": ["% Pases completados", "Pases progresivos", "Acarreos progresivos", "Toques (Ataq. pen.)"],
        "Defensas": ["Derribos", "Intercepciones", "Bloqueos", "Duelos áreos ganados"],
        "Porteros": ['PSxG-GA','Goles en contra','% Paradas','PSxG/SoT', 'Toques', '% Cruces detenidos']

    }

def filtrar_metricas_por_rol(player_info, params_por_rol):
    """
    Filtra las métricas de la Tabla_metricas según el rol_general del jugador y las métricas relevantes definidas
    en params_por_rol. Añade las métricas filtradas como 'metricas_radar' al diccionario del jugador.

    Args:
        player (dict): Diccionario que contiene los datos del jugador.
        params_por_rol (dict): Diccionario que asocia roles generales con métricas específicas.

    Returns:
        dict: Diccionario del jugador actualizado con las métricas filtradas.
    """
    # Obtener el rol general del jugador
    rol = player_info.get('rol_general', None)

    # Verificar que el rol está en params_por_rol
    if rol not in params_por_rol:
        print(f"El rol '{rol}' no está definido en params_por_rol. No se filtrarán métricas.")
        player_info['metricas_radar'] = None
        return 
    
    # Obtener las métricas relevantes para el rol
    metricas_relevantes = params_por_rol[rol]

    # Filtrar la Tabla_metricas según las métricas relevantes
    tabla_metricas = player_info.get('Tabla_metricas', pd.DataFrame())
    metricas_filtradas = tabla_metricas[tabla_metricas['Statistic'].isin(metricas_relevantes)]

    # Guardar las métricas filtradas en el diccionario del jugador
    player_info['metricas_radar'] = metricas_filtradas

    return player_info



def rangos_radar_pizza(values_jugador, values_liga, params):
    

    # Asegurarnos de que las estructuras sean listas
    values_jugador = values_jugador.tolist() if isinstance(values_jugador, pd.Series) else values_jugador
    values_liga = values_liga.tolist() if isinstance(values_liga, pd.Series) else values_liga

    # Verificar que las listas tengan la misma longitud
    if len(params) == len(values_jugador) == len(values_liga):
        
        # Crear listas dinámicamente para min_range y max_range con márgenes ajustados
        min_range = []
        max_range = []

        colchon_min = 1  # Colchón inferior
        colchon_max =1  # Colchón superior

        for index, param in enumerate(params):  # Iterar usando el índice directamente
            try:
                # Valores de jugador y liga
                jugador_value = float(values_jugador[index])
                liga_value = float(values_liga[index])
                
                # Cálculo de mínimo y máximo
                min_value = min(jugador_value, liga_value)
                max_value = max(jugador_value, liga_value)
                
                # Ajustar los valores respetando el colchón
                min_value_adjusted = max(0, math.floor(min_value) - colchon_min)  # Mínimo con colchón inferior
                max_value_adjusted = math.ceil(max_value) + colchon_max  # Máximo con colchón superior
                
                # Agregar valores a las listas
                min_range.append(min_value_adjusted)
                max_range.append(max_value_adjusted)
            except Exception as e:
                print(f"Error procesando el índice {index}: {e}")
                min_range.append(None)
                max_range.append(None)

    else:
        raise ValueError("Las listas `params`, `values_jugador` y `values_liga` deben tener la misma longitud.")# Verificar si las longitudes coinciden
    
    return min_range, max_range


def generar_radar_pizza(player_info, temporada="2024-25", params=None, min_range=None, max_range=None, values_jugador=None, values_liga=None):
    # Comprobar que los argumentos necesarios no sean None
    if params is None or min_range is None or max_range is None or values_jugador is None or values_liga is None:
        raise ValueError("Los argumentos 'params', 'min_range', 'max_range', 'values_jugador' y 'values_liga' son obligatorios.")

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,
        min_range=min_range,        # min range values
        max_range=max_range,        # max range values
        background_color="#323232", straight_line_color="#000000",
        last_circle_color="#000000", last_circle_lw=3, other_circle_lw=0,
        other_circle_color="#000000", straight_line_lw=1
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values_jugador,                     # list of values
        compare_values=values_liga,    # passing comparison values
        figsize=(9, 9),             # adjust figsize according to your need
        color_blank_space="same",   # use same color to fill blank space
        blank_alpha=0.4,            # alpha for blank-space colors
        param_location=109,         # where the parameters will be added
        kwargs_slices=dict(
            facecolor="#1A78CF", edgecolor="#000000",
            zorder=1, linewidth=1
        ),                          # values to be used when plotting slices
        kwargs_compare=dict(
            facecolor="#ff9300", edgecolor="#222222", zorder=3, linewidth=1,
        ),                          # values to be used when plotting comparison slices
        kwargs_params=dict(
            color="#F2F2F2", fontsize=10, zorder=5,
            fontproperties=font_normal.prop, va="center"
        ),                          # values to be used when adding parameter
        kwargs_values=dict(
            color="#000000", fontsize=10,
            fontproperties=font_normal.prop, zorder=2,
            bbox=dict(
                edgecolor="#000000", facecolor="#1A78CF",
                boxstyle="round,pad=0.2", lw=1
            )
        ),                           # values to be used when adding parameter-values
        kwargs_compare_values=dict(
            color="#000000", fontsize=10,
            fontproperties=font_normal.prop, zorder=2,
            bbox=dict(
                edgecolor="#000000", facecolor="#FF9300",
                boxstyle="round,pad=0.1", lw=1
            )
        )                            # values to be used when adding comparison-values
    )

    # Definir los textos
    nombre_jugador = f"{player_info['Player']}"
    nombre_liga = "League Average"

    # add subtitle
    fig.text(
            0.515, 0.99,
            f"Estadísticas 90s vs Liga Profesional Argentina ({player_info['rol_general']}) \nBasado en {player_info['Minutos_jugados_reporte']} jugados | Temporada: {temporada}",
            size=15,
            ha="center", fontproperties=font_bold.prop, color="#F2F2F2"
        )
    
        # add text for nombre_jugador
    fig.text(
        0.26, 0.95, f"{nombre_jugador}", size=14,  # Fija posición izquierda
        fontproperties=font_bold.prop, color="white", ha="left"
    )

    # add text for nombre_liga
    fig.text(
        0.83, 0.95, f"{nombre_liga}", size=14,  # Fija posición derecha
        fontproperties=font_bold.prop, color="white", ha="right"
    )

        # add rectangles
    fig.patches.extend([
            plt.Rectangle(
                (0.23, 0.95), 0.025, 0.021, fill=True, color="#1A78CF",
                transform=fig.transFigure, figure=fig
            ),
            
            plt.Rectangle(
                (0.63, 0.95), 0.025, 0.021, fill=True, color="#FF9300",
                transform=fig.transFigure, figure=fig
            ),
        ])
    plt.show()

    # Guardar el gráfico como imagen temporal
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
        fig.savefig(temp_img_file.name, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Cerrar el gráfico para liberar memoria

    return temp_img_file.name

def generar_y_procesar_informe(player_info, modelo="llama3:8b"):
        """
        Genera un informe detallado sobre un jugador, usando el modelo de Ollama para procesar los datos.
        
        Args:
            player_data (dict): Diccionario con los datos del jugador. Debe incluir:
                - 'Player': Nombre del jugador
                - 'Position': Posición
                - 'Age': Edad
                - 'Team': Equipo actual
                - 'Similar_players': Tabla de jugadores similares (pandas DataFrame)
                - 'Tabla_metricas': Tabla de métricas avanzadas (pandas DataFrame)
            modelo (str): Nombre del modelo de Ollama. Por defecto es "llama3:8b".
            
        Returns:
            dict: Diccionario con las secciones del informe procesadas:
                - 'Resumen': Texto del resumen
                - 'Fortalezas': Lista de fortalezas
                - 'Debilidades': Lista de debilidades
                - 'Potencial': Texto del potencial
                - 'Jugadores Similares': Lista de jugadores similares
        """
        # Crear el prompt
        prompt_final = f"""
        Actúa como un scout de fútbol profesional con experiencia técnica y táctica.
        Necesito que crees un informe detallado sobre {player_info['Player']}, un jugador de {player_info['Position']} de {player_info['Age']} años que juega en {player_info['Team']}.
        Las secciones del informe serán: Resumen, Fortalezas, Debilidades, Potencial y Jugadores Similares.
        Basado en las estadísticas avanzadas de la tabla {player_info['Tabla_metricas'].to_markdown()} y los datos de todos los demás jugadores similares en {player_info['Similar_players'].to_markdown()},
        proporciona el informe con el siguiente formato:

        **Resumen**: Descripción breve del estilo de juego de {player_info['Player']} y su impacto en el campo.

        **Fortalezas**: En cada fortaleza añade la estadistica del per 90 y el percentil correspondiente. Siempre nombra una lista de 4 fortalezas.
        - Fortaleza 1
        - Fortaleza 2
        - Fortaleza 3
        - Fortaleza 4

        **Debilidades**: En cada debilidad añade la estadistica del per 90 y el percentil correspondiente. Siempre nombra una lista de 4 debilidades.
        - Debilidad 1
        - Debilidad 2
        - Debilidad 3
        - Debilidad 4

        **Potencial**: Evaluación del potencial de desarrollo y áreas de mejora.

        **Jugadores Similares**:
        - Jugador 1: Descripción y equipo
        - Jugador 2: Descripción y equipo
        - Jugador 3: Descripción y equipo
        - Jugador 4: Descripción y equipo
        - Jugador 5: Descripción y equipo
        """

        # Llamada al modelo de Ollama
        response = ollama.chat(model=modelo, messages=[{'role': 'user', 'content': prompt_final}])
        contenido = response.get('message', {}).get('content', "")
        if not contenido:
            raise ValueError("El modelo Ollama no devolvió una respuesta válida. Verifica la conexión o el prompt.")
        
        # Diccionario para almacenar las partes procesadas
        partes = {
            "Resumen": "",
            "Fortalezas": [],
            "Debilidades": [],
            "Potencial": "",
            "Jugadores Similares": []
        }
        
        # Expresiones regulares para las secciones "Resumen" y "Potencial"
        secciones_regex = {
            "Resumen": r"\*\*Resumen\*\*:(.*?)(?=\*\*|$)",  # Captura todo después de **Resumen** hasta el siguiente encabezado
            "Potencial": r"\*\*Potencial\*\*:(.*?)(?=\*\*|$)"
        }
        
        # Extraer las secciones "Resumen" y "Potencial" usando regex
        for seccion, regex in secciones_regex.items():
            resultado = re.search(regex, contenido, re.DOTALL)  # .DOTALL permite que el punto capture saltos de línea
            if resultado:
                partes[seccion] = resultado.group(1).strip()  # Extraer y limpiar el contenido
        
        # Función auxiliar para extraer listas (Fortalezas, Debilidades y Jugadores Similares)
        def extraer_lista(texto):
            items = re.findall(r"\* (.*?)\n", texto)  # Captura el texto después del asterisco y hasta el salto de línea
            return [item.strip() for item in items]  # Limpiar los elementos de la lista

        # Procesar las fortalezas, debilidades y jugadores similares
        if "**Fortalezas**" in contenido:
            fortalezas_texto = contenido.split("**Fortalezas**")[1].split("**Debilidades**")[0]
            partes["Fortalezas"] = extraer_lista(fortalezas_texto)

        if "**Debilidades**" in contenido:
            debilidades_texto = contenido.split("**Debilidades**")[1].split("**Potencial**")[0]
            partes["Debilidades"] = extraer_lista(debilidades_texto)

        if "**Jugadores Similares**" in contenido:
            jugadores_similares_texto = contenido.split("**Jugadores Similares**")[1]
            partes["Jugadores Similares"] = extraer_lista(jugadores_similares_texto)
        
        # Si alguna sección está vacía, colocar un mensaje predeterminado
        for seccion in ["Resumen", "Potencial"]:
            if not partes[seccion]:
                partes[seccion] = "Información no proporcionada."
        
        # Aquí ya tienes todo procesado. Puedes desestructurarlo
        resumen_info = partes["Resumen"]
        fortalezas_info = partes["Fortalezas"]
        debilidades_info = partes["Debilidades"]
        potencial_info = partes["Potencial"]
        jugadores_similares_info = partes["Jugadores Similares"]

        # Si necesitas devolver el diccionario completo, lo puedes hacer
        return resumen_info ,fortalezas_info , debilidades_info, potencial_info,jugadores_similares_info

# Función para filtrar jugadores según la información dada
def filtrar_jugadores(df, jugadores_similares_info, jugador_principal):
        """
        Filtra jugadores basado en nombres y asegura que el jugador principal esté incluido.
        """
        # Si jugadores_similares_info es una lista, extraer los nombres de los jugadores
        nombres_jugadores = []
        if isinstance(jugadores_similares_info, list):
            for descripcion in jugadores_similares_info:
                # Buscar nombres entre los asteriscos (por ejemplo, **Sebastián Villa**)
                nombres = re.findall(r'\*\*(.*?)\*\*', descripcion)
                if nombres:  # Si se encontraron nombres
                    nombres_jugadores.extend(nombres)

        # Asegurarse de que el jugador principal esté en la lista de nombres
        if jugador_principal not in nombres_jugadores:
            nombres_jugadores.append(jugador_principal)

        # Filtrar el DataFrame por los nombres de los jugadores
        return df[df['Player'].isin(nombres_jugadores)].drop_duplicates('Player')

    # Función para procesar las columnas del DataFrame
def procesar_columnas(df, columnas_base, stats_plot):
        """
        Procesa un DataFrame con las siguientes operaciones:
        1. Filtra las columnas de interés.
        2. Elimina columnas duplicadas.
        3. Convierte las columnas de porcentaje a números decimales.
        4. Elimina el símbolo '+' y convierte las columnas con valores de tipo +/- a float.
        """
        # Filtrar columnas de interés
        columnas_interes = columnas_base + stats_plot["Statistic"]
        df5 = df[columnas_interes]
        
        # Eliminar las primeras columnas y resetear el índice
        df6 = df5[columnas_interes[2:]].reset_index(drop=True)
        
        # Eliminar columnas duplicadas
        df_procesado = df6.loc[:, ~df6.columns.duplicated()]
        
        # Eliminar el símbolo '%' y convertir las columnas de porcentaje a números decimales
        if 'keepers_Save%' in df_procesado.columns:
            df_procesado['keepers_Save%'] = df_procesado['keepers_Save%'].str.replace('%', '')
        if 'keepersadv_Stp%' in df_procesado.columns:
            df_procesado['keepersadv_Stp%'] = df_procesado['keepersadv_Stp%'].str.replace('%', '')
        
        # Eliminar el signo '+' y convertir la columna 'keepersadv_PSxG+/-' a float
        if 'keepersadv_PSxG+/-' in df_procesado.columns:
            df_procesado['keepersadv_PSxG+/-'] = df_procesado['keepersadv_PSxG+/-'].str.replace('+', '').astype(float)
        
        return df_procesado

    # Función para normalizar las estadísticas
def normalizar_por_90s(df, normalize_column, exclude_columns):
        """
        Normaliza columnas de un DataFrame dividiéndolas por una columna específica, excluyendo otras,
        y redondea los valores resultantes a dos decimales.

        Args:
            df (pd.DataFrame): El DataFrame de entrada.
            normalize_column (str): La columna por la que se normalizarán los valores.
            exclude_columns (list): Lista de columnas a excluir del proceso.

        Returns:
            pd.DataFrame: El DataFrame con las columnas normalizadas y redondeadas a 2 decimales.
        """
        # Copiar el DataFrame para evitar modificar el original
        df_normalized = df.copy()
        
        # Asegurarse de que la columna de normalización sea numérica
        df_normalized[normalize_column] = pd.to_numeric(df_normalized[normalize_column], errors='coerce')
        
        # Verificar si hay valores nulos en la columna de normalización
        if df_normalized[normalize_column].isnull().any():
            raise ValueError(f"La columna '{normalize_column}' contiene valores nulos o no numéricos.")
        
        # Seleccionar columnas que se normalizarán (excluir las especificadas)
        columns_to_normalize = [col for col in df_normalized.columns if col not in exclude_columns + [normalize_column]]
        
        # Normalizar las columnas seleccionadas y redondear a 2 decimales
        for col in columns_to_normalize:
            # Convertir la columna a numérica si no lo es
            df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')

            # Aplicar la normalización y redondear
            df_normalized[col] = (df_normalized[col] / df_normalized[normalize_column]).round(2)
        
        return df_normalized

    # Función para ejecutar todo el flujo completo, dependiendo de si es portero o no
def flujo_completo(liga, año, jugadores_similares_info, jugador_principal, player_data):
        """
        Ejecuta todo el flujo dependiendo de si el jugador es un portero o un jugador de campo.
        """
        # Detectar automáticamente si el jugador es portero
        is_gk = player_data.get("Position") == "GK"
        
        # Scraping dependiendo de la posición del jugador (GK o jugador de campo)
        if is_gk:
            print(f"El jugador {jugador_principal} es portero. Procesando estadísticas de portero...")
            df_players = fbref.get_all_player_season_stats(liga, año)[1]  # Índice 1 para porteros
            stats_plot = {
                "Statistic": [
                    "keepersadv_PSxG+/-", "keepers_GA", "keepers_Save%", "keepersadv_Stp%",
                    "keepersadv_PSxG/SoT"
                ]
            }
            stats_labels = {
                "keepersadv_PSxG+/-": "PSxG-GA",
                "keepers_GA": "Goles encajados",
                "keepers_Save%": '% Paradas', "keepersadv_Stp%": '% Centros detenidos',
                "keepersadv_PSxG/SoT": "PSxG/SoT",
            }
            # Columnas base específicas para porteros
            columnas_base = ['Player', 'keepers_Squad', 'keepers_90s']
            
        else:
            print(f"El jugador {jugador_principal} NO es portero. Procesando estadísticas de jugador de campo...")
            df_players = fbref.get_all_player_season_stats(liga, año)[0]  # Índice 0 para jugadores de campo
            stats_plot = {
                "Statistic": [
                    "passing_CrsPA", "gca_GCA90", "defense_Mid 3rd", "defense_Sh", "defense_Int",
                    "possession_Touches", "possession_Att Pen", "possession_CPA", "misc_Recov"
                ]
            }
            stats_labels = {
                "passing_CrsPA": "Centros al área de penalti",
                "gca_GCA90": "Acciones creadas de gol",
                "defense_Mid 3rd": "Entradas (mid 3rd)",
                "defense_Sh": "Tiros bloqueados",
                "defense_Int": "Intercepciones",
                "possession_Touches": "Toques",
                "possession_Att Pen": "Toques intentados área de penalti",
                "possession_CPA": "Progresiones al área rival",
                "misc_Recov": "Recuperaciones"
            }
            # Columnas base específicas para jugadores de campo
            columnas_base = ['Player', 'stats_Squad', 'stats_90s']
        
        # Columnas de interés
        columnas_interes = columnas_base + stats_plot["Statistic"]
        
        # Validar scraping
        if df_players.empty:
            raise ValueError(f"No se obtuvieron datos de la liga '{liga}' y el año '{año}'. Verifica el scraping.")
        
    
        # Validar si las columnas base existen en el DataFrame
        columnas_base_faltantes = [col for col in columnas_base if col not in df_players.columns]
        if columnas_base_faltantes:
            raise ValueError(f"Las columnas base faltantes en el DataFrame son: {columnas_base_faltantes}")
        
        # Filtrar jugadores
        df_filtrado = filtrar_jugadores(df_players, jugadores_similares_info, jugador_principal)
        
        # Verificar si hay datos tras el filtrado
        if df_filtrado.empty:
            raise ValueError(f"No se encontraron datos para el jugador '{jugador_principal}' en el DataFrame.")
        
        # Procesar estadísticas
        df_procesado = procesar_columnas(df_filtrado, columnas_base, stats_plot)
        
        # Normalizar estadísticas por 'stats_90s' o 'keepers_90s' dependiendo de si es portero
        normalize_column = 'keepers_90s' if is_gk else 'stats_90s'
        df_normalizado = normalizar_por_90s(
            df_procesado,
            normalize_column=normalize_column,
            exclude_columns=["keepersadv_PSxG/SoT", "keepers_Save%", "keepersadv_Stp%","gca_GCA90"]
        )
        
        # Renombrar columnas con las etiquetas
        df_final = df_normalizado.rename(columns=stats_labels)
        # Asegurarse de que ambos DataFrames tienen el mismo índice
        df_filtrado = df_filtrado.reset_index(drop=True)

        # Agregar las columnas 'Player' y 'Team' al final del DataFrame final
        df_final['Player'] = df_filtrado['Player'].values
        # Usar 'keepers_Squad' si es un portero, sino usar 'stats_Squad'
        df_final['stats_Squad'] = df_filtrado['keepers_Squad'].values if is_gk else df_filtrado['stats_Squad'].values

        # Convertir 'Player' y 'stats_Squad' a tipo string
        df_final['Player'] = df_final['Player'].astype(str)
        df_final['stats_Squad'] = df_final['stats_Squad'].astype(str)

        columnas_a_convertir = [col for col in df_final.columns if col not in ['Player', 'stats_Squad']]
        
        # Convertir las columnas seleccionadas a valores numéricos, forzando NaN para errores
        for col in columnas_a_convertir:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

        columnas_actuales= df_final.columns.tolist()
        # Reordenamos las columnas para que 'Player' y 'stats_Squad' estén al principio
        nuevas_columnas = ['Player', 'stats_Squad'] + [col for col in columnas_actuales if col not in ['Player', 'stats_Squad']]
        # Aplicamos el nuevo orden al DataFrame
        df_final = df_final[nuevas_columnas]
        return df_final
   

def configurar_jugadores_y_valores(df, columnas_valores_inicio=3, player_data=  {"Position": "GK"}):
        """Extrae jugadores, equipos, valores, y define parámetros del radar."""
        # Definir los parámetros del radar según la posición del jugador (portero o jugador de campo)
        # Detectar automáticamente si el jugador es portero
        es_portero = player_data.get("Position") == "GK"
        if es_portero:
            params = [
                "PSxG-GA", "Goles encajados", "% Paradas", "% Centros detenidos", "PSxG/SoT"
            ]
        else:
            params = [
                "Centros al área de penalti", "Acciones creadas de gol", "Entradas (mid 3rd)", 
                "Tiros bloqueados", "Intercepciones", "Toques", "Toques intentados área de penalti",  
                "Progresiones al área rival", "Recuperaciones"
            ]
        jugadores = df['Player'].tolist()   
        equipos = df['stats_Squad'].tolist()
        
        jugadores_valores = {row['Player']: row[columnas_valores_inicio:].tolist() for _, row in df.iterrows()}
        
        low1 = df.iloc[:, columnas_valores_inicio:].min().apply(np.floor) - 5
        
        high1 = df.iloc[:, columnas_valores_inicio:].max().apply(np.ceil) + 1
        
        radar = Radar(params, low1.clip(lower=0).tolist(), high1.tolist(),
                    lower_is_better=['Miscontrol'],  # Ajusta si 'Miscontrol' es relevante
                    round_int=[False]*len(params),
                    num_rings=4,  # el número de anillos
                    ring_width=1, center_circle_radius=1)
        return jugadores, equipos, jugadores_valores, low1.clip(lower=0).tolist(), high1.tolist(), params, radar

def extraer_nombre_jugador(descripcion):
        """
        Extrae el nombre del jugador de una descripción y limpia los asteriscos y otros símbolos innecesarios.
        """
        # Eliminar asteriscos y posibles espacios extra antes y después del nombre
        nombre_limpio = re.sub(r'^\*+|\*+$', '', descripcion.split(":")[0]).strip()
        return nombre_limpio

def radar_chart_comparativo(jugador_1, jugador_2, radar, jugadores_valores, jugadores, equipos, df,
                                color_jugador_1='#01c49d', color_jugador_2='#d80499',
                                fontsize_labels=20, figheight=12):
        """
        Genera un radar comparativo entre dos jugadores.
        """
        try:
            # Verificar existencia de jugadores en la lista de jugadores
            if jugador_1 not in jugadores or jugador_2 not in jugadores:
                raise ValueError(f"Uno o ambos jugadores no están en la lista de jugadores.")
            
            # Valores de los jugadores
            jugador_1_valores = jugadores_valores[jugador_1]
            jugador_2_valores = jugadores_valores[jugador_2]
            
        
            # Verificar si las longitudes coinciden
            if len(jugador_1_valores) != len(jugador_2_valores):
                raise ValueError(f"Los jugadores no tienen la misma cantidad de valores. "
                                f"{jugador_1} tiene {len(jugador_1_valores)} valores, "
                                f"{jugador_2} tiene {len(jugador_2_valores)} valores.")
        
            # Obtener los índices de los jugadores en la lista de jugadores
            index_jugador_1 = jugadores.index(jugador_1)
            index_jugador_2 = jugadores.index(jugador_2)
            
            # Obtener los equipos de los jugadores desde la lista 'equipos'
            equipo_jugador_1 = equipos[index_jugador_1]
            equipo_jugador_2 = equipos[index_jugador_2]
            
        
            # Crear figura
            fig, axs = grid(figheight=figheight, grid_height=0.915, title_height=0.06,
                            endnote_height=0.025, title_space=0, endnote_space=0,
                            grid_key='radar', axis=False)
            
            # Configurar radar
            radar.setup_axis(ax=axs['radar'], facecolor='None')
            radar.draw_circles(ax=axs['radar'], facecolor='#28252c', edgecolor='#39353f', lw=1.5)
            
            # Dibujar radar comparativo
            radar_output = radar.draw_radar_compare(
                jugador_1_valores, jugador_2_valores, ax=axs['radar'],
                kwargs_radar={'facecolor': color_jugador_1, 'alpha': 0.6},
                kwargs_compare={'facecolor': color_jugador_2, 'alpha': 0.6}
            )
            radar_poly, radar_poly2, vertices1, vertices2 = radar_output
            
            # Etiquetas y rangos
            radar.draw_range_labels(ax=axs['radar'], fontsize=fontsize_labels, color='#fcfcfc')
            radar.draw_param_labels(ax=axs['radar'], fontsize=fontsize_labels, color='#fcfcfc')
            
            # Puntos en los vértices
            axs['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
                                c=color_jugador_1, edgecolors='#6d6c6d', marker='o', s=150, zorder=2)
            axs['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
                                c=color_jugador_2, edgecolors='#6d6c6d', marker='o', s=150, zorder=2)
            
            # Títulos
            axs['title'].text(0.01, 0.65, jugador_1, fontsize=25, color=color_jugador_1, ha='left', va='center')
            axs['title'].text(0.01, 0.30, equipo_jugador_1, fontsize=15, color=color_jugador_1, ha='left', va='center')
            axs['title'].text(0.99, 0.65, jugador_2, fontsize=25, color=color_jugador_2, ha='right', va='center')
            axs['title'].text(0.99, 0.30, equipo_jugador_2, fontsize=15, color=color_jugador_2, ha='right', va='center')
            fig.set_facecolor('#323232')
            return fig
        except Exception as e:
            print(f"Error al generar el radar: {e}")
            return None

def generar_radares_automaticos(jugador_principal, jugadores_similares_info, radar, jugadores_valores, jugadores, equipos, df,
                                    color_principal='#01c49d', color_similar='#d80499', 
                                    fontsize_labels=20, figheight=12):
        """
        Genera automáticamente radares comparativos entre el jugador principal y cada jugador en jugadores_similares_info.
        Almacena los radares generados en una lista para acceso posterior.
        """
        radares_generados = []  # Lista para guardar las figuras generadas
        
        # Limpiar los nombres de los jugadores similares
        jugadores_similares_limpios = [extraer_nombre_jugador(desc) for desc in jugadores_similares_info]
            
        # Iterar sobre los jugadores similares
        for jugador_similar_desc in jugadores_similares_info:
            jugador_similar = extraer_nombre_jugador(jugador_similar_desc)
            
            if jugador_similar not in jugadores:
                print(f"Jugador similar '{jugador_similar}' no está en la lista de jugadores.")
                continue
            
            print(f"Generando radar para: {jugador_principal} vs {jugador_similar}")
            
            # Crear radar comparativo
            fig = radar_chart_comparativo(
                jugador_1=jugador_principal, 
                jugador_2=jugador_similar, 
                radar=radar, 
                jugadores_valores=jugadores_valores, 
                jugadores=jugadores, 
                equipos=equipos, 
                df=df,
                color_jugador_1=color_principal, 
                color_jugador_2=color_similar, 
                fontsize_labels=fontsize_labels, 
                figheight=figheight
            )
            
            if fig is not None:
                # Guardar la figura en la lista de radares generados
                radares_generados.append(fig)
            else:
                print(f"No se pudo generar el radar para {jugador_principal} vs {jugador_similar}")
            
        
        return radares_generados


def generar_pdf(link):
    player_info = extract_player_info(link)
    player_info= get_player_with_stats(player_info['Report_URL_liga_argentina'])
    pdf = PDF(player_name=player_info['Player'])
    pdf.add_page()

    # Información del jugador
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(255, 255, 255)  # Texto en blanco
    pdf.ln(3)
    pdf.cell(200, 10, f"Jugador: {player_info['Player']}", ln=True)
    pdf.cell(200, 10, f"Posición: {player_info['Position']}", ln=True)
    pdf.cell(200, 10, f"Edad: {player_info['Age']}", ln=True)
    pdf.cell(200, 10, f"Equipo: {player_info['Team']}", ln=True)
       
    # URL de la imagen del jugador
    original_img_path = None  
    # URL de la imagen del jugador
    img_url = player_info.get('Photo_URL', "")  # Si no hay URL, se asigna una cadena vacía

    try:
        if img_url:  # Si la URL existe (no está vacía)
            # Descargar la imagen original
            response = requests.get(img_url)
            if response.status_code == 200:
                original_img = Image.open(BytesIO(response.content))

                # Guardar la imagen original temporalmente para agregarla al PDF
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                    original_img.save(temp_img_file, format="PNG")
                    original_img_path = temp_img_file.name

                # Eliminar el fondo solo si la URL es válida
                img_no_bg_path = removeBackground(img_url)  # Procesa la imagen para quitar el fondo
                if img_no_bg_path:
                    pdf.image(img_no_bg_path, x=158, y=30, w=35, h=35)  # Añade la imagen sin fondo al PDF
                else:
                    print("Error al procesar la imagen sin fondo.")
            else:
                print(f"Error al descargar la imagen: {img_url}, código de estado: {response.status_code}")
        else:
            # Si no hay URL de la imagen, usar la imagen predeterminada
            pdf.image("assets/default-removebg-preview.png", x=158, y=30, w=35, h=35)  # Añadir la imagen predeterminada al PDF

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")

    #EJECUTAR PROMPT CON IA
    
    resumen_info ,fortalezas_info , debilidades_info, potencial_info,jugadores_similares_info= generar_y_procesar_informe(player_info)
    print(resumen_info)
    print(fortalezas_info)
    print(debilidades_info)
    print(potencial_info)
    print(jugadores_similares_info)
    # Resumen (centrado)
    pdf.ln(1)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(255, 255, 255)  # Texto en blanco
    pdf.cell(0, 10, "RESUMEN:", ln=True, align="C")  # Centrando el título de Resumen
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, resumen_info)
    pdf.ln(3)

    # Generar el gráfico de radar
    values_liga= obtener_valores_liga (player_info, especifico_a_general,df_valores_jugadores_delanteros, df_valores_jugadores_centrocampistas, df_valores_jugadores_defensas, df_valores_gk )
    player_info = filtrar_metricas_por_rol(player_info, params_por_rol)
    

    # Verificar el resultado
    values_jugador= player_info['metricas_radar']['Per 90']
    values_jugador = values_jugador.tolist()
    params1= player_info['metricas_radar']['Statistic'].tolist()


    min_range, max_range= rangos_radar_pizza(values_jugador,values_liga,params1)
    radar_img_path = generar_radar_pizza( player_info=player_info, temporada="2024-25", params=params1, min_range=min_range, max_range=max_range,
                            values_jugador=values_jugador,  values_liga=values_liga)
    
    # Agregar el gráfico de radar al PDF
    pdf.chapter_title("ESTADÍSTICAS 90s RESPECTO A LA LIGA PROFESIONAL ARGENTINA")
    pdf.image(radar_img_path, x=43, y=pdf.get_y(), w=135)  # Ajustar la posición y tamaño según sea necesario# Agregar el gráfico de radar pizza
    pdf.ln(5)

    pdf.add_page()
    # Fortalezas - Color verde
    pdf.ln(1)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 128, 0)  # Verde
    pdf.cell(200, 10, "FORTALEZAS:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(255, 255, 255)  # Texto de fortalezas en blanco
    
    # Aquí se aplica el formato correcto de la lista
    for item in fortalezas_info:
        if ":" in item:
            nombre, descripcion = item.split(":", 1)
            nombre = nombre.strip().replace("**", "")
            descripcion = descripcion.strip()

            # Lista con nombre en negrita y descripción normal
            pdf.set_font("Arial", "B", 10)  # Negrita para el nombre
            pdf.multi_cell(0, 8, f"- {nombre}:")  # Mostrar el nombre en negrita con un guion delante

            # Descripción en formato normal
            pdf.set_font("Arial", "", 10)  # Normal para la descripción
            pdf.set_x(13)  # Mueve el cursor 15 unidades hacia la derecha
            pdf.multi_cell(0, 8, descripcion)  # Mostrar la descripción en formato normal

    # Debilidades - Color rojo
    pdf.ln(1)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(255, 0, 0)  # Rojo
    pdf.cell(200, 10, "DEBILIDADES:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(255, 255, 255)  # Texto de debilidades en blanco
    
    # Aquí se aplica el formato correcto de la lista para debilidades
    for item in debilidades_info:
        if ":" in item:
            nombre, descripcion = item.split(":", 1)
            nombre = nombre.strip().replace("**", "")
            descripcion = descripcion.strip()

            # Lista con nombre en negrita y descripción normal
            pdf.set_font("Arial", "B", 10)  # Negrita para el nombre
            pdf.multi_cell(0, 8, f"- {nombre}:")  # Mostrar el nombre en negrita con un guion delante

            # Descripción en formato normal
            pdf.set_font("Arial", "", 10)  # Normal para la descripción
            pdf.set_x(13)  # Mueve el cursor 15 unidades hacia la derecha
            pdf.multi_cell(0, 8, descripcion)  # Mostrar la descripción en formato normal

    # Potencial (centrado) - Color dorado
    pdf.ln(2)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(218, 165, 32)  # Dorado
    pdf.cell(0, 10, "POTENCIAL:", ln=True, align="C")  # Centrando el título de Potencial
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(255, 255, 255)  # Texto del potencial en blanco
    pdf.set_x(13)  # Mueve el cursor 15 unidades hacia la derecha
    pdf.multi_cell(0, 8, potencial_info)

    pdf.add_page()
    # Jugadores Similares - Mantener texto en blanco
    pdf.ln(3)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(255, 255, 255)  # Texto en blanco
    pdf.cell(200, 10, "JUGADORES SIMILARES:", ln=True, align="C")
    pdf.set_font("Arial", "", 12)

    # Aquí se aplica el formato correcto de la lista para jugadores similares
    for item in jugadores_similares_info:
        if ":" in item:
            nombre, descripcion = item.split(":", 1)
            nombre = nombre.strip().replace("**", "")
            descripcion = descripcion.strip()

            # Lista con nombre en negrita y descripción normal
            pdf.set_font("Arial", "B", 10)  # Negrita para el nombre
            pdf.multi_cell(0, 8, f"- {nombre}:")  # Mostrar el nombre en negrita con un guion delante

            # Descripción en formato normal
            pdf.set_font("Arial", "", 10)  # Normal para la descripción
            pdf.set_x(13)  # Mueve el cursor 15 unidades hacia la derecha
            pdf.multi_cell(0, 8, descripcion)  # Mostrar la descripción en formato normal
    
   # Ejecutar el flujo completo
    df_final = flujo_completo(liga="Primera Division Argentina", año="2024", jugadores_similares_info=jugadores_similares_info,   
                            jugador_principal=player_info['Player'], player_data=player_info)
   # Llamada a la función para configurar jugadores y valores
    jugadores, equipos, jugadores_valores, low1, high1, params, radar = configurar_jugadores_y_valores(df_final, columnas_valores_inicio=3, player_data=player_info)


    radares_generados = generar_radares_automaticos(jugador_principal=player_info['Player'], jugadores_similares_info=jugadores_similares_info,
    radar=radar, jugadores_valores=jugadores_valores, jugadores=jugadores, equipos=equipos, df=df_final)

    pdf.add_page()
    pdf.chapter_title2("COMPARATIVA JUGADORES SIMILARES-LIGA PROFESIONAL ARGENTINA")  
    # Dimensiones para las imágenes y espacio entre ellas
    col_width = 90  # Ancho de cada columna
    row_height = 40  # Altura de cada fila
    x_start_left = 10  # Posición x para la columna izquierda
    x_start_right = 110  # Posición x para la columna derecha
    y_margin = 3  # Margen inicial en y
    max_columns = 2  # Número máximo de columnas por fila

    # Variables para controlar posición
    x_positions = [x_start_left, x_start_right]  # Columnas: izquierda y derecha
    y_position = pdf.get_y() + y_margin  # Comienza desde la posición actual del PDF

    # Iterar sobre cada radar generado
    for i, radar_fig in enumerate(radares_generados):
        if radar_fig is None:
            print(f"Error: El radar no es válido.")
            continue

        # Crear un archivo temporal para cada imagen de radar
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            # Guardar la figura como una imagen en formato PNG en el archivo temporal
            radar_fig.savefig(temp_file, format='PNG', bbox_inches='tight')
            temp_file_path = temp_file.name  # Obtiene la ruta del archivo temporal

        # Determinar columna y fila
        col_index = i % max_columns  # Índice de columna (0 = izquierda, 1 = derecha)
        x_position = x_positions[col_index]  # Posición x basada en la columna

        # Si es una nueva fila (primer radar de la fila), ajusta la posición y
        if col_index == 0 and i > 0:
            y_position += row_height + 25  # Incrementa la posición y (espaciado entre filas)

        # Agregar la imagen al PDF
        pdf.image(temp_file_path, x=x_position, y=y_position, w=col_width)

        # Si hemos llenado una fila (ya tenemos 2 imágenes), aumenta la posición Y para la siguiente fila
        if col_index == 1:  # Si ya hemos añadido una imagen en la columna derecha
            y_position += row_height   # Incrementa la posición y para la siguiente fila

        # Salto final de línea si es necesario
    pdf.ln(y_position + row_height)
    

    # Guardar el PDF final
    filename = f"assets/smart_report_{player_info['Player']}.pdf"
    pdf.output(filename)

    # Eliminar el archivo temporal solo si original_img_path tiene un valor válido
    if original_img_path and os.path.exists(original_img_path):
        os.remove(original_img_path)

    return True, filename
