"""
Preprocesamiento y tokenización del corpus
- Limpieza y normalización
- Tokenización con SentencePiece
- Generación de vocabularios
- Split train/valid/test
"""
import sys
import re
import pandas as pd
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm
import json

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

class TextCleaner:
    """Limpieza y normalización de texto"""
    
    @staticmethod
    def clean_text(text):
        """Aplica todas las transformaciones de limpieza"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Normalizar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar espacios al inicio/final
        text = text.strip()
        
        return text
    
    @staticmethod
    def is_valid_pair(src, tgt, min_len=config.MIN_LENGTH, max_len=config.MAX_LENGTH):
        """
        Valida si un par de oraciones cumple los criterios:
        - Longitud mínima y máxima
        - No vacías
        - Proporción razonable entre source y target
        """
        src_words = src.split()
        tgt_words = tgt.split()
        
        src_len = len(src_words)
        tgt_len = len(tgt_words)
        
        # Verificar longitudes
        if src_len < min_len or src_len > max_len:
            return False
        if tgt_len < min_len or tgt_len > max_len:
            return False
        
        # Verificar proporción (evitar traducciones muy asimétricas)
        ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
        if ratio > 2.5:
            return False
        
        return True

def load_and_clean_corpus():
    """Carga y limpia el corpus paralelo"""
    print("📖 Cargando corpus paralelo...")
    
    corpus_file = config.RAW_DATA_DIR / "parallel_corpus.tsv"
    
    if not corpus_file.exists():
        raise FileNotFoundError(
            f"❌ Corpus no encontrado en {corpus_file}\n"
            f"   Ejecuta primero: python src/download_data.py"
        )
    
    # Cargar datos
    df = pd.read_csv(corpus_file, sep='\t', encoding='utf-8')
    print(f"   Pares cargados: {len(df):,}")
    
    # Limpiar y filtrar
    print("🧹 Limpiando y filtrando pares...")
    cleaner = TextCleaner()
    
    cleaned_pairs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src = cleaner.clean_text(str(row['source']))
        tgt = cleaner.clean_text(str(row['target']))
        
        if cleaner.is_valid_pair(src, tgt):
            cleaned_pairs.append((src, tgt))
    
    print(f"   Pares válidos después de filtrado: {len(cleaned_pairs):,}")
    print(f"   Descartados: {len(df) - len(cleaned_pairs):,} "
          f"({100 * (len(df) - len(cleaned_pairs)) / len(df):.1f}%)")
    
    return cleaned_pairs

def train_tokenizer(texts, lang, vocab_size=config.VOCAB_SIZE):
    """
    Entrena un tokenizer SentencePiece para un idioma
    
    Args:
        texts: Lista de textos para entrenamiento
        lang: Código de idioma ('es' o 'en')
        vocab_size: Tamaño del vocabulario
    """
    print(f"\n🔤 Entrenando tokenizer SentencePiece para '{lang}'...")
    
    # Ajustar vocab_size según el tamaño del corpus
    # Regla general: vocab_size <= num_sentences / 4
    max_vocab = max(100, len(texts) // 4)
    if vocab_size > max_vocab:
        vocab_size = max_vocab
        print(f"   ⚠️  Vocabulario ajustado a {vocab_size} (corpus pequeño)")
    
    # Guardar textos temporalmente para SentencePiece
    temp_file = config.TOKENIZER_DIR / f"temp_{lang}.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    # Configuración del tokenizer
    model_prefix = config.TOKENIZER_DIR / f"spm_{lang}"
    
    spm.SentencePieceTrainer.train(
        input=str(temp_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',  # Byte Pair Encoding
        pad_id=config.PAD_IDX,
        bos_id=config.BOS_IDX,
        eos_id=config.EOS_IDX,
        unk_id=config.UNK_IDX,
        pad_piece=config.PAD_TOKEN,
        bos_piece=config.BOS_TOKEN,
        eos_piece=config.EOS_TOKEN,
        unk_piece=config.UNK_TOKEN,
    )
    
    # Eliminar archivo temporal
    temp_file.unlink()
    
    print(f"   ✅ Tokenizer guardado: {model_prefix}.model")
    
    # Cargar y verificar
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    print(f"   📊 Tamaño de vocabulario: {sp.get_piece_size()}")
    
    # Ejemplo de tokenización
    if texts:
        example = texts[0]
        tokens = sp.encode(example, out_type=str)
        ids = sp.encode(example, out_type=int)
        print(f"   📝 Ejemplo: '{example}'")
        print(f"      Tokens: {tokens[:10]}...")
        print(f"      IDs: {ids[:10]}...")
    
    return sp

def split_data(pairs, train_ratio=config.TRAIN_SPLIT, 
               valid_ratio=config.VALID_SPLIT, 
               test_ratio=config.TEST_SPLIT):
    """Divide los datos en train/valid/test"""
    import random
    random.seed(config.SEED)
    
    # Mezclar datos
    pairs = list(pairs)
    random.shuffle(pairs)
    
    n = len(pairs)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    
    train_pairs = pairs[:train_end]
    valid_pairs = pairs[train_end:valid_end]
    test_pairs = pairs[valid_end:]
    
    print(f"\n📊 División de datos:")
    print(f"   Train: {len(train_pairs):,} pares ({100*len(train_pairs)/n:.1f}%)")
    print(f"   Valid: {len(valid_pairs):,} pares ({100*len(valid_pairs)/n:.1f}%)")
    print(f"   Test:  {len(test_pairs):,} pares ({100*len(test_pairs)/n:.1f}%)")
    
    return train_pairs, valid_pairs, test_pairs

def save_processed_data(train_pairs, valid_pairs, test_pairs, 
                        sp_src, sp_tgt):
    """
    Guarda los datos procesados y tokenizados
    Formato: JSON Lines para fácil lectura posterior
    """
    print(f"\n💾 Guardando datos procesados...")
    
    for split_name, pairs in [('train', train_pairs), 
                               ('valid', valid_pairs), 
                               ('test', test_pairs)]:
        
        output_file = config.PROCESSED_DATA_DIR / f"{split_name}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for src, tgt in tqdm(pairs, desc=f"Guardando {split_name}"):
                # Tokenizar a IDs
                src_ids = sp_src.encode(src, out_type=int)
                tgt_ids = sp_tgt.encode(tgt, out_type=int)
                
                # Guardar en formato JSON
                data = {
                    'source_text': src,
                    'target_text': tgt,
                    'source_ids': src_ids,
                    'target_ids': tgt_ids
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"   ✅ {split_name}: {output_file}")
    
    # Guardar metadatos
    metadata = {
        'source_lang': config.SOURCE_LANG,
        'target_lang': config.TARGET_LANG,
        'vocab_size': config.VOCAB_SIZE,
        'train_size': len(train_pairs),
        'valid_size': len(valid_pairs),
        'test_size': len(test_pairs),
        'min_length': config.MIN_LENGTH,
        'max_length': config.MAX_LENGTH,
    }
    
    metadata_file = config.PROCESSED_DATA_DIR / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Metadatos: {metadata_file}")

def get_corpus_statistics(train_pairs, valid_pairs, test_pairs):
    """Calcula y muestra estadísticas detalladas del corpus"""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DEL CORPUS PROCESADO")
    print("="*60)
    
    all_pairs = train_pairs + valid_pairs + test_pairs
    
    src_lengths = [len(src.split()) for src, _ in all_pairs]
    tgt_lengths = [len(tgt.split()) for _, tgt in all_pairs]
    
    print(f"\n📦 Total de pares: {len(all_pairs):,}")
    print(f"\n📏 Longitudes (en palabras):")
    print(f"   Source (español):")
    print(f"      Media: {sum(src_lengths)/len(src_lengths):.2f}")
    print(f"      Min: {min(src_lengths)}, Max: {max(src_lengths)}")
    print(f"   Target (inglés):")
    print(f"      Media: {sum(tgt_lengths)/len(tgt_lengths):.2f}")
    print(f"      Min: {min(tgt_lengths)}, Max: {max(tgt_lengths)}")
    
    print(f"\n🌐 Idiomas: {config.SOURCE_LANG} → {config.TARGET_LANG}")
    print(f"📚 Tamaño de vocabulario: {config.VOCAB_SIZE}")
    print(f"🔤 Tokenizer: SentencePiece (BPE)")
    
    print("="*60)

def main():
    """Pipeline completo de preprocesamiento"""
    print("🚀 Iniciando preprocesamiento del corpus...\n")
    
    # 1. Cargar y limpiar
    pairs = load_and_clean_corpus()
    
    # 2. Dividir en train/valid/test
    train_pairs, valid_pairs, test_pairs = split_data(pairs)
    
    # 3. Entrenar tokenizers
    train_src = [src for src, _ in train_pairs]
    train_tgt = [tgt for _, tgt in train_pairs]
    
    sp_src = train_tokenizer(train_src, config.SOURCE_LANG)
    sp_tgt = train_tokenizer(train_tgt, config.TARGET_LANG)
    
    # 4. Guardar datos procesados
    save_processed_data(train_pairs, valid_pairs, test_pairs, sp_src, sp_tgt)
    
    # 5. Estadísticas finales
    get_corpus_statistics(train_pairs, valid_pairs, test_pairs)
    
    print("\n✅ ¡Preprocesamiento completado!")
    print(f"📁 Datos procesados en: {config.PROCESSED_DATA_DIR}")
    print(f"🔤 Tokenizers en: {config.TOKENIZER_DIR}")

if __name__ == "__main__":
    main()
