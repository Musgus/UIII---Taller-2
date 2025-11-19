"""Entrena únicamente el modelo Transformer con la configuración actual."""

import argparse
import sys
from pathlib import Path

# Habilitar imports locales
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # pylint: disable=wrong-import-position
from src.dataset import create_dataloaders, load_tokenizers  # pylint: disable=wrong-import-position
from src.model_transformer import create_transformer_model  # pylint: disable=wrong-import-position
from src.train import Trainer  # pylint: disable=wrong-import-position
from src.utils import set_seed  # pylint: disable=wrong-import-position


def run_training(num_epochs: int, resume: bool) -> None:
    set_seed(config.SEED)

    train_loader, valid_loader, _ = create_dataloaders(batch_size=config.BATCH_SIZE)
    src_tokenizer, tgt_tokenizer = load_tokenizers()

    model = create_transformer_model(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
    )

    learning_rate = config.TRANSFORMER_CONFIG.get("learning_rate", config.LEARNING_RATE)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model_name=config.TRANSFORMER_CONFIG["name"],
        learning_rate=learning_rate,
    )

    if resume:
        checkpoint_path = config.MODELS_DIR / config.TRANSFORMER_CONFIG["name"] / "last_checkpoint.pt"
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"⚠️  No se encontró checkpoint en {checkpoint_path}, se entrenará desde cero.")

    trainer.train(num_epochs=num_epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar solo el Transformer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Número de épocas a entrenar",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar desde last_checkpoint.pt si existe",
    )
    args = parser.parse_args()

    run_training(num_epochs=args.epochs, resume=args.resume)


if __name__ == "__main__":
    main()
