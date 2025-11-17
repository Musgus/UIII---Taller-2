"""
Descarga y preparación del dataset OPUS Tatoeba (es-en)
Este script descarga automáticamente el corpus paralelo y lo prepara para entrenamiento.
"""
import sys
import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

def download_file(url, destination):
    """Descarga un archivo con barra de progreso"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_tatoeba_dataset():
    """
    Descarga el dataset Tatoeba español-inglés desde OPUS
    https://opus.nlpl.eu/Tatoeba.php
    """
    print("📥 Descargando dataset Tatoeba (es-en) desde OPUS...")
    
    # URLs del corpus paralelo Tatoeba
    base_url = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/"
    
    # Archivos a descargar
    files = {
        "es-en.txt.zip": f"{base_url}es-en.txt.zip"
    }
    
    for filename, url in files.items():
        dest_path = config.RAW_DATA_DIR / filename
        
        if dest_path.exists():
            print(f"✅ {filename} ya existe, saltando descarga...")
            continue
        
        try:
            print(f"⬇️  Descargando {filename}...")
            download_file(url, dest_path)
            
            # Descomprimir
            if filename.endswith('.zip'):
                print(f"📦 Descomprimiendo {filename}...")
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(config.RAW_DATA_DIR)
                print(f"✅ Descomprimido exitosamente")
                
        except Exception as e:
            print(f"❌ Error descargando {filename}: {e}")
            print(f"💡 Puedes descargar manualmente desde: {url}")
            return False
    
    return True

def prepare_parallel_corpus():
    """
    Prepara el corpus paralelo en formato TSV
    Cada línea: source\ttarget
    """
    print("\n📝 Preparando corpus paralelo...")
    
    # Buscar archivos descargados
    source_file = config.RAW_DATA_DIR / f"Tatoeba.es-en.es"
    target_file = config.RAW_DATA_DIR / f"Tatoeba.es-en.en"
    
    if not source_file.exists() or not target_file.exists():
        print("❌ Archivos fuente no encontrados.")
        print(f"   Esperados: {source_file} y {target_file}")
        print("\n💡 INSTRUCCIONES ALTERNATIVAS:")
        print("   1. Ve a https://opus.nlpl.eu/Tatoeba.php")
        print("   2. Descarga 'moses/es-en.txt.zip'")
        print(f"   3. Extrae los archivos en: {config.RAW_DATA_DIR}")
        return False
    
    # Leer pares paralelos
    with open(source_file, 'r', encoding='utf-8') as f_src, \
         open(target_file, 'r', encoding='utf-8') as f_tgt:
        source_lines = f_src.readlines()
        target_lines = f_tgt.readlines()
    
    # Verificar que tienen la misma cantidad
    assert len(source_lines) == len(target_lines), \
        "❌ Los archivos source y target tienen diferente número de líneas"
    
    # Guardar en formato TSV
    output_file = config.RAW_DATA_DIR / "parallel_corpus.tsv"
    
    print(f"💾 Guardando corpus paralelo en {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source\ttarget\n")
        for src, tgt in tqdm(zip(source_lines, target_lines), 
                              total=len(source_lines),
                              desc="Procesando pares"):
            src = src.strip()
            tgt = tgt.strip()
            if src and tgt:  # Filtrar líneas vacías
                f.write(f"{src}\t{tgt}\n")
    
    print(f"✅ Corpus paralelo guardado: {len(source_lines)} pares")
    return True

def get_dataset_stats():
    """Muestra estadísticas del dataset descargado"""
    corpus_file = config.RAW_DATA_DIR / "parallel_corpus.tsv"
    
    if not corpus_file.exists():
        print("❌ Corpus no encontrado. Ejecuta la descarga primero.")
        return
    
    print("\n📊 Estadísticas del Dataset:")
    print("=" * 50)
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header
    
    num_pairs = len(lines)
    
    src_lengths = []
    tgt_lengths = []
    
    for line in lines[:10000]:  # Muestra de 10k para estadísticas rápidas
        try:
            src, tgt = line.strip().split('\t')
            src_lengths.append(len(src.split()))
            tgt_lengths.append(len(tgt.split()))
        except:
            continue
    
    print(f"📦 Total de pares: {num_pairs:,}")
    print(f"📏 Longitud promedio (source): {sum(src_lengths)/len(src_lengths):.1f} palabras")
    print(f"📏 Longitud promedio (target): {sum(tgt_lengths)/len(tgt_lengths):.1f} palabras")
    print(f"📏 Longitud máxima (source): {max(src_lengths)} palabras")
    print(f"📏 Longitud máxima (target): {max(tgt_lengths)} palabras")
    print("=" * 50)

def main():
    """Pipeline completo de descarga y preparación"""
    print("🚀 Iniciando descarga y preparación de datos...\n")
    
    # Opción 1: Intento automático de descarga
    success = download_tatoeba_dataset()
    
    if success:
        success = prepare_parallel_corpus()
    
    if success:
        get_dataset_stats()
        print("\n✅ ¡Datos listos para preprocessing!")
    else:
        print("\n⚠️  Descarga automática falló.")
        print("\n📋 INSTRUCCIONES MANUALES:")
        print("=" * 50)
        print("1. Visita: https://opus.nlpl.eu/Tatoeba.php")
        print("2. En la sección 'Download', selecciona:")
        print("   - Source language: Spanish (es)")
        print("   - Target language: English (en)")
        print("3. Descarga el archivo 'moses/es-en.txt.zip'")
        print(f"4. Extrae el contenido en: {config.RAW_DATA_DIR}")
        print("5. Ejecuta este script nuevamente")
        print("=" * 50)
        
        # Crear archivo de ejemplo para testing
        print("\n🔧 Creando corpus de ejemplo para testing...")
        create_sample_corpus()

def create_sample_corpus():
    """
    Crea un corpus de ejemplo pequeño para testing
    cuando no se pueden descargar los datos reales
    """
    sample_data = [
        ("Hola, ¿cómo estás?", "Hello, how are you?"),
        ("Buenos días.", "Good morning."),
        ("Me llamo Juan.", "My name is Juan."),
        ("¿Dónde está el baño?", "Where is the bathroom?"),
        ("Muchas gracias.", "Thank you very much."),
        ("De nada.", "You're welcome."),
        ("Hasta luego.", "See you later."),
        ("¿Qué hora es?", "What time is it?"),
        ("No entiendo.", "I don't understand."),
        ("¿Hablas inglés?", "Do you speak English?"),
        ("Necesito ayuda.", "I need help."),
        ("Por favor.", "Please."),
        ("Lo siento.", "I'm sorry."),
        ("¿Cuánto cuesta?", "How much does it cost?"),
        ("Me gusta mucho.", "I like it a lot."),
    ] * 100  # Repetir para tener ~1500 pares
    
    output_file = config.RAW_DATA_DIR / "parallel_corpus.tsv"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source\ttarget\n")
        for src, tgt in sample_data:
            f.write(f"{src}\t{tgt}\n")
    
    print(f"✅ Corpus de ejemplo creado: {len(sample_data)} pares")
    print(f"   Ubicación: {output_file}")
    print("   ⚠️  NOTA: Este es un dataset MÍNIMO para testing.")
    print("            Para resultados reales, usa el dataset completo de OPUS.")

if __name__ == "__main__":
    main()
