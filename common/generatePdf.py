from fpdf import FPDF
from common.removeBackground import removeBackground
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from PIL import Image
from io import BytesIO
import os
import tempfile

class PDF(FPDF):
    def header(self):
        self.set_fill_color(0, 0, 0)  # Fondo negro
        self.rect(0, 0, self.w, self.h, 'F')  # Crear fondo negro

def extract_player_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extraer información básica
    player_info = {}
    player_info['Player'] = soup.find('h1').find('span').text.strip()

    # Extraer la posición del jugador
    position_element = soup.select_one('p:-soup-contains("Position")')
    if position_element:
        player_info['Position'] = position_element.text.split(':')[1].split(' ')[0].strip()
    else:
        player_info['Position'] = None

    # Extraer el equipo del jugador
    paragraphs = soup.find_all('p')
    player_team = None
    for p in paragraphs:
        if p.find('a'):
            player_team = p.find('a').text.strip()
            break
    player_info['Team'] = player_team if player_team else 'Equipo no encontrado'

    # Extraer la fecha de nacimiento y calcular la edad
    birthday = soup.select_one('span[id="necro-birth"]')
    if birthday:
        birthday = birthday.text.strip()
        player_info['Age'] = (datetime.now() - datetime.strptime(birthday, '%B %d, %Y')).days // 365
    else:
        player_info['Age'] = None

    # Extraer la URL de la imagen del jugador
    img_tag = soup.find('div', class_='media-item').find('img')
    if img_tag and 'src' in img_tag.attrs:
        player_info['Photo_URL'] = img_tag['src']
    else:
        player_info['Photo_URL'] = None

    return player_info

def generar_pdf(link):
    player_info = extract_player_info(link)
    pdf = PDF()
    pdf.add_page()

    # Información del jugador
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(255, 255, 255)  # Texto en blanco
    pdf.ln(5)
    pdf.cell(250, 10, f"Jugador: {player_info['Player']}", ln=True)
    pdf.cell(250, 10, f"Posición: {player_info['Position']}", ln=True)
    pdf.cell(250, 10, f"Edad: {player_info['Age']}", ln=True)
    pdf.cell(250, 10, f"Equipo: {player_info['Team']}", ln=True)

    # URL de la imagen del jugador
    img_url = player_info['Photo_URL']
    if not img_url:
        raise ValueError("No se encontró la URL de la imagen del jugador.")

    # Descargar la imagen original con fondo
    response = requests.get(img_url)
    original_img = Image.open(BytesIO(response.content))

    # Guardar la imagen original temporalmente para agregarla al PDF
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
        original_img.save(temp_img_file, format="PNG")
        original_img_path = temp_img_file.name

    # Agregar la imagen original al PDF
    pdf.image(original_img_path, x=50, y=70, w=50, h=50)

    # Obtener la imagen sin fondo usando `removeBackground`
    img_no_bg_path = removeBackground(img_url)

    # Agregar la imagen sin fondo al PDF en otra posición
    if img_no_bg_path:
        pdf.image(img_no_bg_path, x=120, y=70, w=50, h=50)  # Ajusta la posición para que no se superpongan
    else:
        raise ValueError("No se pudo obtener la imagen sin fondo.")

    # Guardar el PDF final
    filename = "assets/smart_report.pdf"
    pdf.output(filename)

    # Eliminar el archivo temporal de la imagen original
    os.remove(original_img_path)

    return True, filename
