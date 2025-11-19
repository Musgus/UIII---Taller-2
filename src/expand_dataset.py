"""
Script para ampliar el dataset sintéticamente
Genera variaciones y combinaciones para aumentar el corpus
"""
import sys
import random
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Plantillas de oraciones para generar corpus aumentado
TEMPLATES = {
    "greetings": [
        ("Hola", "Hello"),
        ("Buenos días", "Good morning"),
        ("Buenas tardes", "Good afternoon"),
        ("Buenas noches", "Good night"),
        ("¿Cómo estás?", "How are you?"),
        ("¿Cómo te llamas?", "What's your name?"),
        ("Mucho gusto", "Nice to meet you"),
        ("Encantado", "Pleased to meet you"),
    ],
    
    "farewells": [
        ("Adiós", "Goodbye"),
        ("Hasta luego", "See you later"),
        ("Hasta pronto", "See you soon"),
        ("Nos vemos", "See you"),
        ("Que tengas un buen día", "Have a good day"),
        ("Cuídate", "Take care"),
    ],
    
    "courtesy": [
        ("Por favor", "Please"),
        ("Gracias", "Thank you"),
        ("Muchas gracias", "Thank you very much"),
        ("De nada", "You're welcome"),
        ("Lo siento", "I'm sorry"),
        ("Disculpa", "Excuse me"),
        ("Perdón", "Pardon me"),
    ],
    
    "questions": [
        ("¿Dónde está el baño?", "Where is the bathroom?"),
        ("¿Qué hora es?", "What time is it?"),
        ("¿Cuánto cuesta?", "How much does it cost?"),
        ("¿Hablas inglés?", "Do you speak English?"),
        ("¿Puedes ayudarme?", "Can you help me?"),
        ("¿Dónde está la estación?", "Where is the station?"),
        ("¿Cómo llego al hotel?", "How do I get to the hotel?"),
    ],
    
    "statements": [
        ("Me llamo Juan", "My name is Juan"),
        ("Vivo en Madrid", "I live in Madrid"),
        ("Soy estudiante", "I am a student"),
        ("Trabajo en una oficina", "I work in an office"),
        ("Me gusta la música", "I like music"),
        ("Estudio español", "I study Spanish"),
        ("Tengo hambre", "I'm hungry"),
        ("Estoy cansado", "I'm tired"),
        ("No entiendo", "I don't understand"),
        ("Necesito ayuda", "I need help"),
    ],
    
    "food": [
        ("Quiero comer", "I want to eat"),
        ("La comida está deliciosa", "The food is delicious"),
        ("Me gusta el café", "I like coffee"),
        ("Quiero agua", "I want water"),
        ("Tengo sed", "I'm thirsty"),
        ("El restaurante es bueno", "The restaurant is good"),
        ("La cena está lista", "Dinner is ready"),
    ],
    
    "travel": [
        ("Necesito un taxi", "I need a taxi"),
        ("¿Dónde está el aeropuerto?", "Where is the airport?"),
        ("Quiero un boleto", "I want a ticket"),
        ("El tren llega pronto", "The train arrives soon"),
        ("El avión sale a las tres", "The plane leaves at three"),
        ("Estoy de vacaciones", "I'm on vacation"),
    ],
    
    "weather": [
        ("Hace buen tiempo", "The weather is nice"),
        ("Está lloviendo", "It's raining"),
        ("Hace frío", "It's cold"),
        ("Hace calor", "It's hot"),
        ("Está nublado", "It's cloudy"),
        ("El sol brilla", "The sun is shining"),
    ],
    
    "numbers": [
        ("Tengo veinte años", "I'm twenty years old"),
        ("Son las cinco", "It's five o'clock"),
        ("Cuesta diez euros", "It costs ten euros"),
        ("Hay tres personas", "There are three people"),
        ("Tengo dos hermanos", "I have two siblings"),
    ],
    
    "actions": [
        ("Voy a la escuela", "I go to school"),
        ("Estoy estudiando", "I am studying"),
        ("Voy a trabajar", "I'm going to work"),
        ("Quiero dormir", "I want to sleep"),
        ("Necesito descansar", "I need to rest"),
        ("Voy a casa", "I'm going home"),
        ("Estoy leyendo un libro", "I'm reading a book"),
        ("Estoy viendo la televisión", "I'm watching TV"),
    ]
}

# Conectores y variaciones
CONNECTORS_ES = ["y", "pero", "porque", "aunque", "cuando", "mientras"]
CONNECTORS_EN = ["and", "but", "because", "although", "when", "while"]

NAMES = ["Juan", "María", "Pedro", "Ana", "Carlos", "Laura", "José", "Carmen"]
PLACES = [
    ("Madrid", "Madrid"),
    ("Barcelona", "Barcelona"),
    ("España", "Spain"),
    ("México", "Mexico"),
    ("la ciudad", "the city"),
    ("el hotel", "the hotel"),
    ("la escuela", "the school"),
    ("el parque", "the park"),
]

def generate_simple_pairs():
    """Genera pares simples de todas las plantillas"""
    pairs = []
    for category, templates in TEMPLATES.items():
        pairs.extend(templates)
    return pairs

def generate_compound_sentences():
    """Genera oraciones compuestas combinando plantillas"""
    pairs = []
    all_templates = []
    for templates in TEMPLATES.values():
        all_templates.extend(templates)
    
    # Generar combinaciones con conectores
    for i in range(500):
        t1 = random.choice(all_templates)
        t2 = random.choice(all_templates)
        conn_idx = random.randint(0, len(CONNECTORS_ES)-1)
        
        es = f"{t1[0].lower()} {CONNECTORS_ES[conn_idx]} {t2[0].lower()}"
        en = f"{t1[1].lower()} {CONNECTORS_EN[conn_idx]} {t2[1].lower()}"
        
        # Capitalizar primera letra
        es = es[0].upper() + es[1:]
        en = en[0].upper() + en[1:]
        
        pairs.append((es, en))
    
    return pairs

def generate_with_names():
    """Genera oraciones con nombres propios"""
    pairs = []
    base_templates = [
        ("Me llamo {name}", "My name is {name}"),
        ("{name} está aquí", "{name} is here"),
        ("Conozco a {name}", "I know {name}"),
        ("{name} es mi amigo", "{name} is my friend"),
        ("Vi a {name} ayer", "I saw {name} yesterday"),
    ]
    
    for template in base_templates:
        for name in NAMES:
            es = template[0].format(name=name)
            en = template[1].format(name=name)
            pairs.append((es, en))
    
    return pairs

def generate_with_places():
    """Genera oraciones con lugares"""
    pairs = []
    base_templates = [
        ("Vivo en {place}", "I live in {place}"),
        ("Voy a {place}", "I'm going to {place}"),
        ("Estoy en {place}", "I'm in {place}"),
        ("{place} es hermoso", "{place} is beautiful"),
        ("Quiero ir a {place}", "I want to go to {place}"),
    ]
    
    for template in base_templates:
        for place_es, place_en in PLACES:
            es = template[0].format(place=place_es)
            en = template[1].format(place=place_en)
            pairs.append((es, en))
    
    return pairs

def generate_numbers():
    """Genera oraciones con números"""
    pairs = []
    numbers_es = ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
                  "once", "doce", "trece", "catorce", "quince", "veinte", "treinta", "cien"]
    numbers_en = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "one hundred"]
    
    for es, en in zip(numbers_es, numbers_en):
        pairs.append((f"Tengo {es} libros", f"I have {en} books"))
        pairs.append((f"Son las {es}", f"It's {en} o'clock"))
        pairs.append((f"Hay {es} personas", f"There are {en} people"))
    
    return pairs

def generate_variations():
    """Genera variaciones de oraciones existentes"""
    pairs = []
    
    # Variaciones de preguntas
    question_variations = [
        ("¿Cómo estás?", "How are you?"),
        ("¿Cómo te encuentras?", "How are you feeling?"),
        ("¿Qué tal?", "How's it going?"),
        ("¿Cómo andas?", "How are you doing?"),
    ]
    
    # Variaciones de despedidas
    farewell_variations = [
        ("Adiós amigo", "Goodbye friend"),
        ("Hasta mañana", "See you tomorrow"),
        ("Nos vemos mañana", "See you tomorrow"),
        ("Hasta la próxima", "Until next time"),
    ]
    
    pairs.extend(question_variations)
    pairs.extend(farewell_variations)
    
    return pairs

def expand_dataset(target_size=20000):
    """
    Genera un dataset expandido con el tamaño objetivo
    
    Args:
        target_size: Número objetivo de pares de oraciones
    """
    print(f"\n🔧 Generando dataset expandido (objetivo: {target_size:,} pares)...\n")
    
    all_pairs = set()  # Usar set para evitar duplicados
    
    # 1. Pares simples
    print("📝 Generando pares simples...")
    simple = generate_simple_pairs()
    all_pairs.update(simple)
    print(f"   ✅ {len(simple)} pares simples")
    
    # 2. Oraciones compuestas
    print("🔗 Generando oraciones compuestas...")
    compound = generate_compound_sentences()
    all_pairs.update(compound)
    print(f"   ✅ {len(compound)} oraciones compuestas")
    
    # 3. Con nombres
    print("👤 Generando oraciones con nombres...")
    with_names = generate_with_names()
    all_pairs.update(with_names)
    print(f"   ✅ {len(with_names)} oraciones con nombres")
    
    # 4. Con lugares
    print("📍 Generando oraciones con lugares...")
    with_places = generate_with_places()
    all_pairs.update(with_places)
    print(f"   ✅ {len(with_places)} oraciones con lugares")
    
    # 5. Con números
    print("🔢 Generando oraciones con números...")
    with_numbers = generate_numbers()
    all_pairs.update(with_numbers)
    print(f"   ✅ {len(with_numbers)} oraciones con números")
    
    # 6. Variaciones
    print("🔄 Generando variaciones...")
    variations = generate_variations()
    all_pairs.update(variations)
    print(f"   ✅ {len(variations)} variaciones")
    
    # 7. Si aún no llegamos al objetivo, generar más combinaciones
    current_size = len(all_pairs)
    if current_size < target_size:
        print(f"\n🎲 Generando combinaciones adicionales ({target_size - current_size} faltantes)...")
        all_templates = []
        for templates in TEMPLATES.values():
            all_templates.extend(templates)
        
        while len(all_pairs) < target_size:
            # Generar combinaciones aleatorias
            t1 = random.choice(all_templates)
            t2 = random.choice(all_templates)
            conn_idx = random.randint(0, len(CONNECTORS_ES)-1)
            
            # Variación 1: Conectar oraciones
            if random.random() < 0.5:
                es = f"{t1[0].lower()} {CONNECTORS_ES[conn_idx]} {t2[0].lower()}"
                en = f"{t1[1].lower()} {CONNECTORS_EN[conn_idx]} {t2[1].lower()}"
            else:
                # Variación 2: Usar solo una plantilla con variaciones
                es = t1[0]
                en = t1[1]
                
                # Agregar palabras adicionales
                if random.random() < 0.3:
                    es += " también"
                    en += " too"
                if random.random() < 0.3:
                    es += " ahora"
                    en += " now"
            
            es = es[0].upper() + es[1:] if es else es
            en = en[0].upper() + en[1:] if en else en
            
            # Asegurar que termina con puntuación
            if es and es[-1] not in '.!?':
                es += '.'
            if en and en[-1] not in '.!?':
                en += '.'
            
            all_pairs.add((es, en))
    
    # Convertir a lista y mezclar
    all_pairs = list(all_pairs)
    random.shuffle(all_pairs)
    
    # Limitar al tamaño objetivo
    all_pairs = all_pairs[:target_size]
    
    # Guardar
    output_file = config.RAW_DATA_DIR / "parallel_corpus.tsv"
    
    print(f"\n💾 Guardando corpus expandido en {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source\ttarget\n")
        for src, tgt in tqdm(all_pairs, desc="Guardando"):
            f.write(f"{src}\t{tgt}\n")
    
    print(f"\n✅ Corpus expandido generado exitosamente!")
    print(f"   📊 Total de pares: {len(all_pairs):,}")
    print(f"   📁 Ubicación: {output_file}")
    print(f"\n⚠️  NOTA: Este es un corpus sintético generado automáticamente.")
    print(f"           Para mejores resultados, usa el dataset real de OPUS Tatoeba (~100k pares).")
    
    return len(all_pairs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expandir dataset para NMT")
    parser.add_argument('--size', type=int, default=20000,
                       help='Número objetivo de pares (default: 20000)')
    args = parser.parse_args()
    
    expand_dataset(target_size=args.size)
