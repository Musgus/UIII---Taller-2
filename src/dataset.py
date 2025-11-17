"""
Dataset personalizado para PyTorch
Maneja la carga eficiente de datos tokenizados
"""
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import sentencepiece as spm
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

class TranslationDataset(Dataset):
    """
    Dataset para traducción automática
    Carga datos pre-tokenizados desde archivos JSONL
    """
    
    def __init__(self, data_file, max_length=config.MAX_LENGTH):
        """
        Args:
            data_file: Ruta al archivo .jsonl (train/valid/test)
            max_length: Longitud máxima de secuencia
        """
        self.data = []
        self.max_length = max_length
        
        # Cargar datos
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                
                # Truncar si es necesario
                src_ids = item['source_ids'][:max_length]
                tgt_ids = item['target_ids'][:max_length]
                
                self.data.append({
                    'source_ids': src_ids,
                    'target_ids': tgt_ids,
                    'source_text': item['source_text'],
                    'target_text': item['target_text']
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retorna un par source-target como tensores"""
        item = self.data[idx]
        
        src_ids = torch.tensor(item['source_ids'], dtype=torch.long)
        tgt_ids = torch.tensor(item['target_ids'], dtype=torch.long)
        
        return {
            'source_ids': src_ids,
            'target_ids': tgt_ids,
            'source_text': item['source_text'],
            'target_text': item['target_text']
        }

def collate_fn(batch):
    """
    Función de collate personalizada para batch padding
    Agrupa múltiples ejemplos y aplica padding dinámico
    """
    # Separar componentes
    source_ids = [item['source_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    source_texts = [item['source_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    
    # Aplicar padding (pad_sequence agrega PAD_IDX por defecto)
    source_ids_padded = pad_sequence(
        source_ids, 
        batch_first=True, 
        padding_value=config.PAD_IDX
    )
    
    target_ids_padded = pad_sequence(
        target_ids, 
        batch_first=True, 
        padding_value=config.PAD_IDX
    )
    
    # Crear máscaras (True donde hay padding)
    source_mask = (source_ids_padded == config.PAD_IDX)
    target_mask = (target_ids_padded == config.PAD_IDX)
    
    # Calcular longitudes reales (útil para pack_padded_sequence)
    source_lengths = torch.tensor([len(seq) for seq in source_ids], dtype=torch.long)
    target_lengths = torch.tensor([len(seq) for seq in target_ids], dtype=torch.long)
    
    return {
        'source_ids': source_ids_padded,
        'target_ids': target_ids_padded,
        'source_mask': source_mask,
        'target_mask': target_mask,
        'source_lengths': source_lengths,
        'target_lengths': target_lengths,
        'source_texts': source_texts,
        'target_texts': target_texts
    }

def create_dataloaders(batch_size=config.BATCH_SIZE, num_workers=0):
    """
    Crea DataLoaders para train, valid y test
    
    Returns:
        train_loader, valid_loader, test_loader
    """
    print("📚 Creando DataLoaders...")
    
    # Crear datasets
    train_dataset = TranslationDataset(
        config.PROCESSED_DATA_DIR / "train.jsonl"
    )
    valid_dataset = TranslationDataset(
        config.PROCESSED_DATA_DIR / "valid.jsonl"
    )
    test_dataset = TranslationDataset(
        config.PROCESSED_DATA_DIR / "test.jsonl"
    )
    
    print(f"   Train: {len(train_dataset):,} ejemplos")
    print(f"   Valid: {len(valid_dataset):,} ejemplos")
    print(f"   Test:  {len(test_dataset):,} ejemplos")
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, valid_loader, test_loader

class Tokenizer:
    """
    Wrapper para SentencePiece con métodos de utilidad
    """
    
    def __init__(self, model_path):
        """
        Args:
            model_path: Ruta al modelo .model de SentencePiece
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
    
    def encode(self, text, add_bos=True, add_eos=True):
        """
        Convierte texto a IDs
        
        Args:
            text: Texto a tokenizar
            add_bos: Agregar token de inicio
            add_eos: Agregar token de fin
        """
        ids = self.sp.encode(text, out_type=int)
        
        if add_bos:
            ids = [config.BOS_IDX] + ids
        if add_eos:
            ids = ids + [config.EOS_IDX]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """
        Convierte IDs a texto
        
        Args:
            ids: Lista o tensor de IDs
            skip_special_tokens: Omitir tokens especiales
        """
        # Convertir tensor a lista si es necesario
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        
        # Filtrar tokens especiales si se solicita
        if skip_special_tokens:
            special_ids = {config.PAD_IDX, config.BOS_IDX, 
                          config.EOS_IDX, config.UNK_IDX}
            ids = [id for id in ids if id not in special_ids]
        
        return self.sp.decode(ids)
    
    def batch_decode(self, batch_ids, skip_special_tokens=True):
        """Decodifica un batch de IDs"""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    @property
    def vocab_size(self):
        """Tamaño del vocabulario"""
        return self.sp.get_piece_size()

def load_tokenizers():
    """
    Carga los tokenizers de source y target
    
    Returns:
        src_tokenizer, tgt_tokenizer
    """
    src_tokenizer = Tokenizer(
        config.TOKENIZER_DIR / f"spm_{config.SOURCE_LANG}.model"
    )
    tgt_tokenizer = Tokenizer(
        config.TOKENIZER_DIR / f"spm_{config.TARGET_LANG}.model"
    )
    
    print(f"🔤 Tokenizers cargados:")
    print(f"   Source ({config.SOURCE_LANG}): vocab_size={src_tokenizer.vocab_size}")
    print(f"   Target ({config.TARGET_LANG}): vocab_size={tgt_tokenizer.vocab_size}")
    
    return src_tokenizer, tgt_tokenizer

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar tokenizers
    src_tokenizer, tgt_tokenizer = load_tokenizers()
    
    # Crear dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(batch_size=4)
    
    # Mostrar un batch de ejemplo
    print("\n📦 Ejemplo de batch:")
    for batch in train_loader:
        print(f"   Source shape: {batch['source_ids'].shape}")
        print(f"   Target shape: {batch['target_ids'].shape}")
        print(f"   Source mask shape: {batch['source_mask'].shape}")
        print(f"\n   Primer ejemplo del batch:")
        print(f"   Source: {batch['source_texts'][0]}")
        print(f"   Target: {batch['target_texts'][0]}")
        print(f"   Source IDs: {batch['source_ids'][0][:20].tolist()}...")
        break
