"""Borra artefactos anteriores y entrena todos los modelos de cero."""

import argparse
import shutil
import sys
from pathlib import Path

# Permitir imports locales
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # pylint: disable=wrong-import-position
from src.dataset import create_dataloaders, load_tokenizers  # pylint: disable=wrong-import-position
from src.model_rnn import create_rnn_model  # pylint: disable=wrong-import-position
from src.model_lstm_attention import create_lstm_model  # pylint: disable=wrong-import-position
from src.model_gru_attention import create_gru_model  # pylint: disable=wrong-import-position
from src.model_transformer import create_transformer_model  # pylint: disable=wrong-import-position
from src.train import Trainer  # pylint: disable=wrong-import-position
from src.utils import set_seed  # pylint: disable=wrong-import-position


MODEL_BUILDERS = {
    config.RNN_CONFIG["name"]: (create_rnn_model, config.RNN_CONFIG),
    config.LSTM_CONFIG["name"]: (create_lstm_model, config.LSTM_CONFIG),
    config.GRU_CONFIG["name"]: (create_gru_model, config.GRU_CONFIG),
    config.TRANSFORMER_CONFIG["name"]: (create_transformer_model, config.TRANSFORMER_CONFIG),
}


def clean_outputs() -> None:
    """Elimina artefactos previos para comenzar desde cero."""
    for directory in [config.MODELS_DIR, config.METRICS_DIR, config.LOGS_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
            print(f"🧹 Eliminado: {directory}")
        directory.mkdir(parents=True, exist_ok=True)


def train_model(model_name: str, epochs: int, patience: int,
                train_loader, valid_loader,
                src_vocab_size: int, tgt_vocab_size: int) -> None:
    create_fn, cfg = MODEL_BUILDERS[model_name]
    learning_rate = cfg.get("learning_rate", config.LEARNING_RATE)

    print(f"\n{'#' * 70}\n# Entrenando {model_name} desde cero\n{'#' * 70}")

    model = create_fn(src_vocab_size, tgt_vocab_size)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model_name=model_name,
        learning_rate=learning_rate,
    )

    trainer.train(num_epochs=epochs, early_stopping_patience=patience)


def main() -> None:
    parser = argparse.ArgumentParser(description="Borrar artefactos y entrenar todos los modelos")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Épocas por modelo (default: config.NUM_EPOCHS)")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE,
                        help="Paciencia para early stopping")
    parser.add_argument("--skip-clean", action="store_true",
                        help="No borrar artefactos antes de entrenar")
    parser.add_argument("--only", nargs="*", choices=list(MODEL_BUILDERS.keys()),
                        help="Entrenar únicamente estos modelos")
    args = parser.parse_args()

    set_seed(config.SEED)

    if not args.skip_clean:
        clean_outputs()
    else:
        print("⚠️  Conservando artefactos existentes (skip-clean activado)")

    train_loader, valid_loader, _ = create_dataloaders(batch_size=config.BATCH_SIZE)
    src_tokenizer, tgt_tokenizer = load_tokenizers()

    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size

    target_models = args.only if args.only else list(MODEL_BUILDERS.keys())

    for model_name in target_models:
        train_model(
            model_name=model_name,
            epochs=args.epochs,
            patience=args.patience,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
        )

    print("\n✅ Entrenamiento completo.")
    print(f"📁 Modelos guardados en: {config.MODELS_DIR}")
    print(f"📁 Métricas en: {config.METRICS_DIR}")


if __name__ == "__main__":
    main()
