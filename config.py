"""
Configuración centralizada para el proyecto NMT
Todos los hiperparámetros y rutas en un solo lugar
"""
import torch
from pathlib import Path

# ============================================
# RUTAS DEL PROYECTO
# ============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"

RESULTADOS_DIR = BASE_DIR / "resultados"
PLOTS_DIR = RESULTADOS_DIR / "plots"
METRICS_DIR = RESULTADOS_DIR / "metrics"

# Crear directorios si no existen
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TOKENIZER_DIR, 
                  MODELS_DIR, LOGS_DIR, PLOTS_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURACIÓN DE DATOS
# ============================================
SOURCE_LANG = "es"
TARGET_LANG = "en"

# Límites de longitud de oraciones
MIN_LENGTH = 3
MAX_LENGTH = 50

# Tamaño de vocabulario para SentencePiece
VOCAB_SIZE = 16000

# Tokens especiales
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ============================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Hyperparameters generales
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP = 1.0

# Split de datos
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1

# ============================================
# ARQUITECTURAS - HIPERPARÁMETROS
# ============================================

# MODELO 1: RNN Simple (sin atención)
RNN_CONFIG = {
    "name": "RNN_Simple",
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.3,
    "use_attention": False
}

# MODELO 2: LSTM con Atención Bahdanau
LSTM_CONFIG = {
    "name": "LSTM_Attention",
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "use_attention": True,
    "attention_type": "bahdanau"
}

# MODELO 3: GRU con Atención Bahdanau
GRU_CONFIG = {
    "name": "GRU_Attention",
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "use_attention": True,
    "attention_type": "bahdanau"
}

# MODELO 4: Transformer
TRANSFORMER_CONFIG = {
    "name": "Transformer",
    "d_model": 256,
    "nhead": 8,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "max_seq_length": MAX_LENGTH,
    "learning_rate": 5e-4,
    "label_smoothing": 0.1,
    "warmup_steps": 400,
    "repetition_penalty": 1.2,
    "top_k": 5,
    "temperature": 1.0
}

# Lista de todas las configuraciones
ALL_CONFIGS = [RNN_CONFIG, LSTM_CONFIG, GRU_CONFIG, TRANSFORMER_CONFIG]

# ============================================
# CONFIGURACIÓN DE EVALUACIÓN
# ============================================
BEAM_WIDTH = 5  # Para beam search durante inferencia
MAX_GENERATE_LENGTH = MAX_LENGTH + 10

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
LOG_INTERVAL = 100  # Cada cuántos batches loggear durante entrenamiento
SAVE_CHECKPOINT_EVERY = 1  # Guardar checkpoint cada N épocas

print(f"🖥️  Dispositivo: {DEVICE}")
print(f"📁 Directorio base: {BASE_DIR}")
print(f"🌐 Par de idiomas: {SOURCE_LANG} → {TARGET_LANG}")
