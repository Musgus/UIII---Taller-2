"""Simple desktop UI for running the trained NMT translators."""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, Optional

import torch


# Ensure local packages are importable
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
	sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

import config  # pylint: disable=wrong-import-position
from src.dataset import load_tokenizers  # pylint: disable=wrong-import-position
from src.model_rnn import create_rnn_model  # pylint: disable=wrong-import-position
from src.model_lstm_attention import create_lstm_model  # pylint: disable=wrong-import-position
from src.model_gru_attention import create_gru_model  # pylint: disable=wrong-import-position
from src.model_transformer import create_transformer_model  # pylint: disable=wrong-import-position
from src.train import load_model_for_inference  # pylint: disable=wrong-import-position


MODEL_REGISTRY = {
	"RNN Simple": {
		"model_name": "RNN_Simple",
		"builder": create_rnn_model,
		"requires_mask": False,
	},
	"LSTM + Atención": {
		"model_name": "LSTM_Attention",
		"builder": create_lstm_model,
		"requires_mask": True,
	},
	"GRU + Atención": {
		"model_name": "GRU_Attention",
		"builder": create_gru_model,
		"requires_mask": True,
	},
	"Transformer": {
		"model_name": "Transformer",
		"builder": create_transformer_model,
		"requires_mask": True,
	},
}


class ModelManager:
	"""Lazy loader for translation models."""

	def __init__(self, vocab_sizes: Dict[str, int], device: torch.device) -> None:
		self.device = device
		self.src_vocab_size = vocab_sizes["src"]
		self.tgt_vocab_size = vocab_sizes["tgt"]
		self.cache: Dict[str, torch.nn.Module] = {}

	def get_model(self, key: str) -> torch.nn.Module:
		if key in self.cache:
			return self.cache[key]

		info = MODEL_REGISTRY[key]
		checkpoint_path = config.MODELS_DIR / info["model_name"] / "best_model.pt"
		if not checkpoint_path.exists():
			raise FileNotFoundError(
				f"No se encontró el checkpoint en {checkpoint_path}. Entrena ese modelo primero."
			)

		model = info["builder"](self.src_vocab_size, self.tgt_vocab_size)
		try:
			model = load_model_for_inference(model, checkpoint_path, device=self.device)
		except RuntimeError as exc:
			msg = str(exc)
			if "size mismatch" in msg:
				raise RuntimeError(
					"El checkpoint encontrado no coincide con el tamaño de vocabulario actual. "
					"Probablemente fue entrenado con las tokenizers antiguas (≈100 tokens). "
					"Vuelve a ejecutar el entrenamiento para ese modelo después de generar las nuevas tokenizers.\n\n"
					f"Detalles técnicos originales:\n{msg}"
				) from exc
			raise

		self.cache[key] = model
		return model


class Translator:
	"""Wrapper around tokenizers and model manager for inference."""

	def __init__(self, device: torch.device) -> None:
		self.device = device
		self.src_tokenizer, self.tgt_tokenizer = load_tokenizers()
		vocab_sizes = {
			"src": self.src_tokenizer.vocab_size,
			"tgt": self.tgt_tokenizer.vocab_size,
		}
		self.model_manager = ModelManager(vocab_sizes, device)

	def available_models(self) -> Dict[str, bool]:
		availability: Dict[str, bool] = {}
		for display_name, info in MODEL_REGISTRY.items():
			checkpoint_path = config.MODELS_DIR / info["model_name"] / "best_model.pt"
			availability[display_name] = checkpoint_path.exists()
		return availability

	def translate(self, text: str, model_key: str) -> str:
		if not text.strip():
			return ""

		model = self.model_manager.get_model(model_key)
		info = MODEL_REGISTRY[model_key]

		# Tokenize input
		src_ids = self.src_tokenizer.encode(text.strip(), add_bos=True, add_eos=True)
		src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
		src_mask: Optional[torch.Tensor] = None
		if info["requires_mask"]:
			src_mask = (src_tensor == config.PAD_IDX)

		with torch.no_grad():
			if info["requires_mask"]:
				generated = model.generate(
					src_tensor,
					src_mask=src_mask,
					max_length=config.MAX_GENERATE_LENGTH,
					device=self.device,
				)
			else:
				generated = model.generate(
					src_tensor,
					max_length=config.MAX_GENERATE_LENGTH,
					device=self.device,
				)

		if isinstance(generated, tuple):
			generated = generated[0]

		output_ids = generated[0].detach().cpu().tolist()
		translation = self.tgt_tokenizer.decode(output_ids, skip_special_tokens=True)
		return translation.strip()


class TranslatorApp:
	"""Tkinter-based UI for quick translations."""

	def __init__(self, root: tk.Tk, translator: Translator) -> None:
		self.root = root
		self.translator = translator
		self.root.title("NMT Translator")

		self.model_var = tk.StringVar()
		self.status_var = tk.StringVar(value="Listo")

		self._build_layout()
		self._populate_models()

	def _build_layout(self) -> None:
		self.root.geometry("720x520")
		self.root.minsize(600, 420)

		main_frame = ttk.Frame(self.root, padding=16)
		main_frame.pack(fill=tk.BOTH, expand=True)

		# Model selection row
		model_frame = ttk.Frame(main_frame)
		model_frame.pack(fill=tk.X, pady=(0, 12))
		ttk.Label(model_frame, text="Modelo:").pack(side=tk.LEFT)
		self.model_combo = ttk.Combobox(
			model_frame,
			textvariable=self.model_var,
			state="readonly",
			width=30,
		)
		self.model_combo.pack(side=tk.LEFT, padx=(8, 0))
		self.model_combo.bind("<<ComboboxSelected>>", lambda _event: self.input_text.focus_set())

		# Input text
		ttk.Label(main_frame, text="Texto origen (Español):").pack(anchor=tk.W)
		self.input_text = tk.Text(main_frame, height=10, wrap=tk.WORD)
		self.input_text.pack(fill=tk.BOTH, expand=True)

		# Buttons
		button_frame = ttk.Frame(main_frame)
		button_frame.pack(fill=tk.X, pady=8)
		translate_btn = ttk.Button(button_frame, text="Traducir", command=self.translate)
		translate_btn.pack(side=tk.LEFT)
		clear_btn = ttk.Button(button_frame, text="Limpiar", command=self.clear)
		clear_btn.pack(side=tk.LEFT, padx=8)

		# Output text
		ttk.Label(main_frame, text="Traducción (Inglés):").pack(anchor=tk.W)
		self.output_text = tk.Text(main_frame, height=10, wrap=tk.WORD, state="disabled")
		self.output_text.pack(fill=tk.BOTH, expand=True)

		# Status bar
		status_bar = ttk.Label(main_frame, textvariable=self.status_var, anchor=tk.W)
		status_bar.pack(fill=tk.X, pady=(12, 0))

		# Key bindings
		self.root.bind("<Control-Return>", lambda _event: self.translate())

	def _populate_models(self) -> None:
		availability = self.translator.available_models()
		available_models = [name for name, is_ready in availability.items() if is_ready]
		if not available_models:
			messagebox.showwarning(
				"Modelos no encontrados",
				"No se encontraron checkpoints en artifacts/models/. Entrena un modelo primero.",
			)
			self.model_combo["values"] = list(availability.keys())
		else:
			self.model_combo["values"] = available_models
			self.model_var.set(available_models[0])

		missing = [name for name, ready in availability.items() if not ready]
		if missing:
			self.status_var.set(
				"Modelos disponibles: "
				+ ", ".join(available_models or ["ninguno"]) + " | Faltan: " + ", ".join(missing)
			)
		else:
			self.status_var.set("Modelos disponibles listos para usar")

	def translate(self) -> None:
		model_key = self.model_var.get()
		text = self.input_text.get("1.0", tk.END)

		if not model_key:
			messagebox.showinfo("Selecciona un modelo", "Por favor elige un modelo de la lista.")
			return

		if not text.strip():
			messagebox.showinfo("Texto vacío", "Escribe una frase para traducir.")
			return

		self.status_var.set(f"Traduciendo con {model_key}...")
		self.root.update_idletasks()

		try:
			translation = self.translator.translate(text, model_key)
		except FileNotFoundError as exc:
			messagebox.showerror("Modelo no disponible", str(exc))
			self.status_var.set("Error: modelo no disponible")
			return
		except Exception as exc:  # pylint: disable=broad-except
			messagebox.showerror("Error", f"Ocurrió un error al traducir:\n{exc}")
			self.status_var.set("Error durante la traducción")
			return

		self._set_output_text(translation or "[Traducción vacía]")
		self.status_var.set("Traducción completada")

	def clear(self) -> None:
		self.input_text.delete("1.0", tk.END)
		self._set_output_text("")
		self.status_var.set("Listo")

	def _set_output_text(self, text: str) -> None:
		self.output_text.configure(state="normal")
		self.output_text.delete("1.0", tk.END)
		self.output_text.insert(tk.END, text)
		self.output_text.configure(state="disabled")


def main() -> None:
	torch.set_grad_enabled(False)
	translator = Translator(device=config.DEVICE)
	root = tk.Tk()
	TranslatorApp(root, translator)
	root.mainloop()


if __name__ == "__main__":
	main()

