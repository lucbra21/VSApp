from rembg import remove
from PIL import Image
import requests
from io import BytesIO
import os

def removeBackground(input_source):
    """
    Elimina el fondo de una imagen
    
    Args:
        input_source: Puede ser una URL o un archivo subido
        
    Returns:
        str: Ruta del archivo de salida
    """
    try:
        output_file = "assets/image_2.png"
        
        # Asegurarse de que el directorio assets existe
        os.makedirs("assets", exist_ok=True)
        
        # Si el input es una URL
        if isinstance(input_source, str) and input_source.startswith('http'):
            response = requests.get(input_source)
            inp = Image.open(BytesIO(response.content))
        # Si es un archivo subido
        else:
            inp = Image.open(input_source)
            
        # Remover el fondo
        out = remove(inp)
        
        # Guardar la imagen
        out.save(output_file)
        return output_file
        
    except Exception as e:
        print(f"Error en removeBackground: {str(e)}")
        return None