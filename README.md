# 🌍 Traducción Automática Neuronal (NMT) - Español → Inglés

Proyecto completo de **Neural Machine Translation (NMT)** que implementa, entrena y compara **4 arquitecturas diferentes** de modelos encoder-decoder para traducción automática de español a inglés.

## 📋 Contenido

- [Características](#-características)
- [Arquitecturas Implementadas](#-arquitecturas-implementadas)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Pipeline Completo](#-pipeline-completo)
- [Resultados](#-resultados)
- [Documentación Técnica](#-documentación-técnica)
- [Extensiones Futuras](#-extensiones-futuras)

## ✨ Características

### 🎯 Modelos Implementados
- ✅ **RNN Simple** (sin atención) - Baseline
- ✅ **LSTM Bidireccional** con Atención Bahdanau
- ✅ **GRU Bidireccional** con Atención Bahdanau
- ✅ **Transformer** (simplificado, 2 capas)

### 🔧 Funcionalidades
- 📦 Descarga automática del dataset OPUS Tatoeba
- 🧹 Preprocesamiento completo (limpieza, normalización, filtrado)
- 🔤 Tokenización subword con **SentencePiece** (BPE)
- 💾 **Checkpointing automático** durante entrenamiento
- 📊 **Métricas persistidas** (BLEU, loss, tiempos, parámetros)
- 📈 **Visualizaciones automáticas** (curvas, comparaciones, tablas)
- 🔄 Early stopping y learning rate scheduling
- 🎨 Análisis detallado por longitud de oración

## 🏗️ Arquitecturas Implementadas

### 1️⃣ RNN Simple (Sin Atención)
**Baseline** - Arquitectura encoder-decoder básica
```
Encoder: Embedding → RNN (2 capas) → Vector contexto
Decoder: Vector contexto → RNN (2 capas) → Linear → Softmax
```
- ⚙️ Parámetros: ~15M
- 📦 Hidden dim: 512
- 🎯 Uso: Comparación baseline

### 2️⃣ LSTM con Atención Bahdanau
**Arquitectura con memoria a largo plazo**
```
Encoder: Embedding → BiLSTM (2 capas) → Outputs
Decoder: Atención(Outputs) + LSTM → Linear → Softmax
```
- ⚙️ Parámetros: ~20M
- 📦 Hidden dim: 512
- 🎯 Atención: Bahdanau (additive)
- 💡 Ventaja: Captura dependencias largas mejor que RNN

### 3️⃣ GRU con Atención Bahdanau
**Similar a LSTM pero más eficiente**
```
Encoder: Embedding → BiGRU (2 capas) → Outputs
Decoder: Atención(Outputs) + GRU → Linear → Softmax
```
- ⚙️ Parámetros: ~18M (25% menos que LSTM)
- 📦 Hidden dim: 512
- 🎯 Atención: Bahdanau (additive)
- 💡 Ventaja: Más rápido que LSTM, similar rendimiento

### 4️⃣ Transformer
**Estado del arte en NMT**
```
Encoder: Embedding + Positional → Multi-Head Self-Attention (2 capas)
Decoder: Embedding + Positional → Masked Self-Attention + Cross-Attention (2 capas)
```
- ⚙️ Parámetros: ~25M
- 📦 d_model: 256, heads: 8
- 🎯 Capas: 2 encoder + 2 decoder
- 💡 Ventaja: Paralelizable, mejor para dependencias largas

## 📁 Estructura del Proyecto

```
Taller 2/
│
├── config.py                      # Configuración centralizada
├── requirements.txt               # Dependencias
├── main.py                        # Script principal (pipeline completo)
│
├── src/                           # Código fuente
│   ├── download_data.py          # Descarga dataset OPUS Tatoeba
│   ├── preprocess.py             # Preprocesamiento y tokenización
│   ├── dataset.py                # PyTorch Dataset y DataLoader
│   ├── model_rnn.py              # Modelo RNN simple
│   ├── model_lstm_attention.py   # Modelo LSTM con atención
│   ├── model_gru_attention.py    # Modelo GRU con atención
│   ├── model_transformer.py      # Modelo Transformer
│   ├── train.py                  # Sistema de entrenamiento
│   ├── evaluate.py               # Evaluación con BLEU
│   ├── visualize.py              # Visualización de resultados
│   └── utils.py                  # Utilidades generales
│
├── data/                          # Datos
│   ├── raw/                      # Corpus crudo
│   │   └── parallel_corpus.tsv
│   └── processed/                # Datos procesados
│       ├── train.jsonl
│       ├── valid.jsonl
│       ├── test.jsonl
│       └── metadata.json
│
├── artifacts/                     # Artefactos del proyecto
│   ├── tokenizer/                # Modelos SentencePiece
│   │   ├── spm_es.model
│   │   └── spm_en.model
│   ├── models/                   # Checkpoints de modelos
│   │   ├── RNN_Simple/
│   │   │   ├── best_model.pt
│   │   │   ├── last_checkpoint.pt
│   │   │   └── architecture.txt
│   │   ├── LSTM_Attention/
│   │   ├── GRU_Attention/
│   │   └── Transformer/
│   └── logs/                     # Logs de entrenamiento
│
└── resultados/                    # Resultados finales
    ├── metrics/                   # Métricas por modelo
    │   ├── RNN_Simple/
    │   │   ├── training_metrics.json
    │   │   ├── evaluation_results.json
    │   │   ├── all_translations.jsonl
    │   │   └── translation_examples.txt
    │   ├── LSTM_Attention/
    │   ├── GRU_Attention/
    │   └── Transformer/
    │
    └── plots/                     # Visualizaciones
        ├── RNN_Simple_training_curves.png
        ├── LSTM_Attention_training_curves.png
        ├── GRU_Attention_training_curves.png
        ├── Transformer_training_curves.png
        ├── all_models_training_comparison.png
        ├── bleu_comparison.png
        ├── bleu_by_length.png
        ├── models_comparison_table.png
        └── training_time_comparison.png
```

## 🚀 Instalación

### Requisitos
- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB+ recomendado)
- GPU con 6GB+ VRAM (opcional pero recomendado)

### Paso 1: Clonar el repositorio
```bash
cd "Taller 2"
```

### Paso 2: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Dependencias principales:
- `torch` - Framework de deep learning
- `sentencepiece` - Tokenización subword
- `sacrebleu` - Cálculo de BLEU
- `matplotlib`, `seaborn` - Visualización
- `tqdm` - Barras de progreso

## 🎯 Uso Rápido

### Opción 1: Pipeline Completo Automático
```bash
# Ejecuta TODO: descarga, preprocesamiento, entrenamiento, evaluación y visualización
python main.py
```

### Opción 2: Paso a Paso

#### 1. Descargar y preparar datos
```bash
python src/download_data.py
```
**Nota**: Si la descarga automática falla, sigue las instrucciones para descarga manual.

#### 2. Preprocesar datos
```bash
python src/preprocess.py
```
Genera:
- Corpus limpio y filtrado
- Tokenizers SentencePiece entrenados
- Splits train/valid/test (80/10/10)

#### 3. Entrenar modelos
```bash
python main.py
```

#### 4. Solo evaluar (si ya entrenaste)
```bash
python main.py --skip-training
```

#### 5. Solo visualizar resultados
```bash
python main.py --skip-training --skip-evaluation
```

## 📊 Pipeline Completo

### Fase 1: Preparación de Datos
```python
python src/download_data.py
python src/preprocess.py
```

**Transformaciones aplicadas:**
1. ✅ Limpieza: minúsculas, normalización de espacios
2. ✅ Filtrado: longitud 3-50 palabras, ratio máximo 2.5:1
3. ✅ Tokenización: SentencePiece BPE (vocab_size=16000)
4. ✅ Split: 80% train, 10% valid, 10% test

### Fase 2: Entrenamiento
```python
python main.py
```

**Configuración de entrenamiento:**
- Batch size: 64
- Learning rate: 0.001 (con ReduceLROnPlateau)
- Épocas: 20 (con early stopping patience=5)
- Optimizador: Adam
- Loss: CrossEntropyLoss (ignora padding)
- Gradient clipping: 1.0
- Teacher forcing: 0.5 (con decaimiento exponencial)

**Guardado automático:**
- ✅ Checkpoint cada época
- ✅ Mejor modelo (best_model.pt)
- ✅ Último checkpoint (last_checkpoint.pt)
- ✅ Métricas en JSON

### Fase 3: Evaluación
```python
# Automático en main.py o manual:
python -c "from evaluate import *; # código evaluación"
```

**Métricas calculadas:**
- 📊 BLEU Score global
- 📊 BLEU por longitud (corta/media/larga)
- 📊 Análisis de errores de longitud
- 📝 50 ejemplos de traducción guardados
- 💾 Todas las traducciones en JSONL

### Fase 4: Visualización
```python
# Automático en main.py o manual con visualize.py
```

**Gráficos generados:**
1. Curvas de loss individuales (train/valid)
2. Comparación de loss entre modelos
3. Comparación de BLEU (gráfico de barras)
4. BLEU por longitud de oración
5. Tabla comparativa completa
6. Comparación de tiempos de entrenamiento

## 📈 Resultados Esperados

### Tabla Comparativa (Ejemplo con Tatoeba es-en)

| Modelo | Parámetros | BLEU | Valid Loss | Tiempo | Observaciones |
|--------|-----------|------|------------|--------|---------------|
| **Transformer** | ~25M | **35-45** | 2.5 | 45 min | 🥇 Mejor BLEU |
| **LSTM Attention** | ~20M | 32-42 | 2.8 | 60 min | 🥈 Buen balance |
| **GRU Attention** | ~18M | 31-41 | 2.9 | 50 min | Más rápido que LSTM |
| **RNN Simple** | ~15M | 25-35 | 3.5 | 40 min | Baseline |

**Nota**: Resultados dependen del tamaño del corpus y recursos computacionales.

### BLEU por Longitud de Oración

```
Oraciones Cortas (≤10 palabras):  BLEU ~40-50
Oraciones Medias (11-20 palabras): BLEU ~30-40
Oraciones Largas (>20 palabras):  BLEU ~20-30
```

## 📚 Documentación Técnica

### Dataset: OPUS Tatoeba
- **Fuente**: https://opus.nlpl.eu/Tatoeba.php
- **Par de idiomas**: Español (es) → Inglés (en)
- **Tamaño**: ~100,000 pares de oraciones
- **Características**:
  - Oraciones cortas y medianas
  - Alta calidad
  - Dominio general

### Tokenización: SentencePiece
- **Algoritmo**: Byte Pair Encoding (BPE)
- **Vocab size**: 16,000 tokens
- **Character coverage**: 99.95%
- **Tokens especiales**:
  - `<pad>` (ID: 0)
  - `<bos>` (ID: 1) - Inicio de secuencia
  - `<eos>` (ID: 2) - Fin de secuencia
  - `<unk>` (ID: 3) - Token desconocido

### Métricas de Evaluación

#### BLEU Score (sacreBLEU)
```python
from sacrebleu.metrics import BLEU
bleu = BLEU()
score = bleu.corpus_score(hypotheses, [references])
```
- **Rango**: 0-100
- **Interpretación**:
  - BLEU < 20: Malo
  - BLEU 20-30: Aceptable
  - BLEU 30-40: Bueno
  - BLEU > 40: Muy bueno

### Comparación de Arquitecturas

#### RNN vs LSTM vs GRU
| Característica | RNN | LSTM | GRU |
|---------------|-----|------|-----|
| **Cell state** | No | Sí | No |
| **Puertas** | 0 | 3 (input, forget, output) | 2 (update, reset) |
| **Parámetros** | Menos | Más | Medio |
| **Entrenamiento** | Rápido | Lento | Medio |
| **Vanishing gradient** | Sí | No | No |

#### RNN/LSTM/GRU vs Transformer
| Característica | RNN/LSTM/GRU | Transformer |
|---------------|--------------|-------------|
| **Procesamiento** | Secuencial | Paralelo |
| **Dependencias largas** | Limitado | Excelente |
| **Velocidad entrenamiento** | Lenta | Rápida (con GPU) |
| **Memoria** | Menos | Más |
| **Estado del arte** | No | Sí |

## 🎓 Conceptos Clave

### Atención Bahdanau
Permite al decoder "mirar" diferentes partes del input en cada paso:

```python
score(h_t, h_s) = v^T * tanh(W_1*h_t + W_2*h_s)
attention_weights = softmax(scores)
context = Σ(attention_weights * encoder_outputs)
```

**Ventajas:**
- ✅ Alinea automáticamente source y target
- ✅ Resuelve bottleneck del vector de contexto fijo
- ✅ Visualizable (mapas de atención)

### Teacher Forcing
Durante entrenamiento, usa el ground truth como input del decoder:

```python
if random() < teacher_forcing_ratio:
    decoder_input = target_token  # Ground truth
else:
    decoder_input = predicted_token  # Predicción
```

**Decaimiento exponencial:**
```python
ratio_epoch_t = ratio_inicial * (0.95 ** (epoch - 1))
```

### Positional Encoding (Transformer)
Inyecta información de posición:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## 🔧 Configuración Avanzada

### Modificar Hiperparámetros
Edita `config.py`:

```python
# Cambiar tamaño de batch
BATCH_SIZE = 128  # Default: 64

# Cambiar learning rate
LEARNING_RATE = 0.0005  # Default: 0.001

# Cambiar épocas
NUM_EPOCHS = 30  # Default: 20

# Cambiar vocabulario
VOCAB_SIZE = 32000  # Default: 16000
```

### Cambiar Arquitectura
Para modificar una arquitectura, edita el archivo correspondiente:
- `src/model_rnn.py`
- `src/model_lstm_attention.py`
- `src/model_gru_attention.py`
- `src/model_transformer.py`

Ejemplo (aumentar capas de Transformer):
```python
TRANSFORMER_CONFIG = {
    "d_model": 512,           # Default: 256
    "num_encoder_layers": 4,  # Default: 2
    "num_decoder_layers": 4,  # Default: 2
}
```

## 🐛 Troubleshooting

### Error: Out of Memory (GPU)
```bash
# Reducir batch size en config.py
BATCH_SIZE = 32  # o menos

# O usar acumulación de gradientes
```

### Error: Dataset no encontrado
```bash
# Descargar manualmente desde:
# https://opus.nlpl.eu/Tatoeba.php
# Colocar en: data/raw/
```

### Error: CUDA not available
```bash
# Verificar instalación de PyTorch con CUDA:
python -c "import torch; print(torch.cuda.is_available())"

# Instalar PyTorch con CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Entrenamiento muy lento
```bash
# 1. Verificar que esté usando GPU
python -c "import config; print(config.DEVICE)"

# 2. Reducir complejidad del modelo
# Editar config.py: reducir hidden_dim, num_layers

# 3. Entrenar con menos datos
# Editar preprocess.py: tomar subset del corpus
```

## 🚀 Extensiones Futuras

### 1. Mejoras de Modelo
- [ ] Beam search (actualmente solo greedy)
- [ ] Label smoothing
- [ ] Byte-level BPE
- [ ] Modelos pre-entrenados (mBART, mT5)

### 2. Más Pares de Idiomas
```python
# En config.py
SOURCE_LANG = "de"  # Alemán
TARGET_LANG = "en"  # Inglés
```

### 3. Dataset Más Grande
- [ ] OPUS-100 (~55M pares)
- [ ] WMT datasets
- [ ] ParaCrawl

### 4. Técnicas Avanzadas
- [ ] Back-translation
- [ ] Multi-task learning
- [ ] Domain adaptation
- [ ] Low-resource NMT

### 5. Deployment
- [ ] API REST con FastAPI
- [ ] Modelo ONNX para inferencia
- [ ] Cuantización para mobile
- [ ] Docker container

## 📖 Referencias

### Papers Fundamentales
1. **Sequence to Sequence Learning** (Sutskever et al., 2014)
2. **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2014)
3. **Attention is All You Need** (Vaswani et al., 2017)

### Recursos Útiles
- [OPUS Corpus](https://opus.nlpl.eu/)
- [SentencePiece](https://github.com/google/sentencepiece)
- [sacreBLEU](https://github.com/mjpost/sacrebleu)
- [PyTorch Seq2Seq Tutorials](https://github.com/bentrevett/pytorch-seq2seq)

## 📝 Licencia

Este proyecto es código educativo de código abierto.

## 👤 Autor

Proyecto desarrollado para el curso de IA III - Traducción Automática Neuronal.

---

## 🎯 Comandos Rápidos

```bash
# Pipeline completo (recomendado)
python main.py

# Solo descarga de datos
python src/download_data.py

# Solo preprocesamiento
python src/preprocess.py

# Entrenar solo un modelo específico
# (editar main.py para comentar otros modelos)

# Evaluar modelos existentes
python main.py --skip-training

# Solo visualizaciones
python main.py --skip-training --skip-evaluation

# Ver información del dispositivo
python -c "from src.utils import print_device_info; print_device_info()"

# Verificar datos preparados
python -c "from main import check_data_ready; check_data_ready()"
```

---

**¡Buena suerte con tu proyecto de traducción automática neuronal! 🚀🌍**
