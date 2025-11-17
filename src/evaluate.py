"""
Evaluación de modelos NMT:
- Cálculo de BLEU score
- Análisis por longitud de oración
- Ejemplos de traducción
- Guardado de resultados
"""
import sys
import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

class NMTEvaluator:
    """
    Evaluador para modelos de traducción automática
    """
    
    def __init__(self, model, test_loader, src_tokenizer, tgt_tokenizer,
                 model_name, device=config.DEVICE):
        
        self.model = model.to(device)
        self.test_loader = test_loader
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.model_name = model_name
        self.device = device
        
        # BLEU scorer
        self.bleu = BLEU()
        
        # Directorio de resultados
        self.results_dir = config.METRICS_DIR / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔍 Evaluador inicializado para {model_name}")
    
    def translate_batch(self, src, src_mask=None):
        """
        Traduce un batch de oraciones
        
        Args:
            src: (batch, src_len)
            src_mask: (batch, src_len)
        
        Returns:
            translations: (batch, max_gen_len) - IDs generados
        """
        self.model.eval()
        
        with torch.no_grad():
            # Llamar al método generate del modelo
            if hasattr(self.model, 'generate'):
                translations = self.model.generate(
                    src, 
                    src_mask=src_mask,
                    max_length=config.MAX_GENERATE_LENGTH,
                    device=self.device
                )
            else:
                raise NotImplementedError(f"Modelo {self.model_name} no tiene método generate")
        
        return translations
    
    def compute_bleu(self, save_translations=True):
        """
        Calcula BLEU score en el conjunto de test
        
        Args:
            save_translations: Si guardar ejemplos de traducción
        
        Returns:
            bleu_score: Score BLEU (0-100)
            results: Diccionario con resultados detallados
        """
        print(f"\n📊 Calculando BLEU score para {self.model_name}...")
        
        self.model.eval()
        
        all_hypotheses = []  # Traducciones generadas
        all_references = []  # Traducciones de referencia
        all_sources = []
        
        # Análisis por longitud
        results_by_length = defaultdict(lambda: {'hyps': [], 'refs': []})
        
        for batch in tqdm(self.test_loader, desc="Traduciendo"):
            src = batch['source_ids'].to(self.device)
            src_mask = batch['source_mask'].to(self.device)
            
            # Generar traducciones
            translations = self.translate_batch(src, src_mask)
            
            # Decodificar a texto
            hyps = self.tgt_tokenizer.batch_decode(
                translations, skip_special_tokens=True
            )
            refs = batch['target_texts']
            srcs = batch['source_texts']
            
            all_hypotheses.extend(hyps)
            all_references.extend(refs)
            all_sources.extend(srcs)
            
            # Clasificar por longitud del source
            for src_text, hyp, ref in zip(srcs, hyps, refs):
                src_len = len(src_text.split())
                if src_len <= 10:
                    length_cat = "short"
                elif src_len <= 20:
                    length_cat = "medium"
                else:
                    length_cat = "long"
                
                results_by_length[length_cat]['hyps'].append(hyp)
                results_by_length[length_cat]['refs'].append(ref)
        
        # Calcular BLEU global
        bleu_score = self.bleu.corpus_score(
            all_hypotheses, 
            [all_references]
        ).score
        
        print(f"\n🎯 BLEU Score: {bleu_score:.2f}")
        
        # Calcular BLEU por longitud
        bleu_by_length = {}
        for length_cat, data in results_by_length.items():
            if data['hyps']:
                score = self.bleu.corpus_score(
                    data['hyps'],
                    [data['refs']]
                ).score
                bleu_by_length[length_cat] = score
                print(f"   BLEU ({length_cat}): {score:.2f} ({len(data['hyps'])} ejemplos)")
        
        # Preparar resultados
        results = {
            'model_name': self.model_name,
            'bleu_score': bleu_score,
            'bleu_by_length': bleu_by_length,
            'num_examples': len(all_hypotheses),
            'avg_hypothesis_length': np.mean([len(h.split()) for h in all_hypotheses]),
            'avg_reference_length': np.mean([len(r.split()) for r in all_references])
        }
        
        # Guardar resultados
        self.save_results(results, all_sources, all_hypotheses, all_references)
        
        if save_translations:
            self.save_translation_examples(
                all_sources, all_hypotheses, all_references, num_examples=50
            )
        
        return bleu_score, results
    
    def save_results(self, results, sources, hypotheses, references):
        """Guarda resultados de evaluación en JSON"""
        results_file = self.results_dir / "evaluation_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Resultados guardados en {results_file}")
        
        # Guardar todas las traducciones
        translations_file = self.results_dir / "all_translations.jsonl"
        
        with open(translations_file, 'w', encoding='utf-8') as f:
            for src, hyp, ref in zip(sources, hypotheses, references):
                data = {
                    'source': src,
                    'hypothesis': hyp,
                    'reference': ref
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"   💾 Traducciones guardadas en {translations_file}")
    
    def save_translation_examples(self, sources, hypotheses, references, num_examples=50):
        """Guarda ejemplos de traducción en formato legible"""
        examples_file = self.results_dir / "translation_examples.txt"
        
        # Seleccionar ejemplos aleatorios
        indices = np.random.choice(
            len(sources), 
            min(num_examples, len(sources)), 
            replace=False
        )
        
        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write(f"EJEMPLOS DE TRADUCCIÓN - {self.model_name}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, idx in enumerate(indices, 1):
                f.write(f"Ejemplo {i}:\n")
                f.write(f"  Fuente:      {sources[idx]}\n")
                f.write(f"  Referencia:  {references[idx]}\n")
                f.write(f"  Traducción:  {hypotheses[idx]}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"   💾 Ejemplos guardados en {examples_file}")
    
    def compare_with_reference(self, source_text, reference_text):
        """
        Traduce una oración individual y compara con referencia
        
        Args:
            source_text: Texto en idioma fuente
            reference_text: Traducción de referencia
        
        Returns:
            hypothesis: Traducción generada
            bleu_score: BLEU de este ejemplo
        """
        self.model.eval()
        
        # Tokenizar
        src_ids = self.src_tokenizer.encode(source_text, add_bos=True, add_eos=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
        src_mask = (src_tensor == config.PAD_IDX)
        
        # Traducir
        with torch.no_grad():
            translation = self.translate_batch(src_tensor, src_mask)
        
        # Decodificar
        hypothesis = self.tgt_tokenizer.decode(
            translation[0], skip_special_tokens=True
        )
        
        # Calcular BLEU
        bleu_score = self.bleu.sentence_score(hypothesis, [reference_text]).score
        
        print(f"\n🔍 Ejemplo de traducción:")
        print(f"   Fuente:      {source_text}")
        print(f"   Referencia:  {reference_text}")
        print(f"   Traducción:  {hypothesis}")
        print(f"   BLEU:        {bleu_score:.2f}")
        
        return hypothesis, bleu_score
    
    def analyze_errors(self, max_examples=100):
        """
        Analiza errores comunes de traducción
        
        Returns:
            error_analysis: Diccionario con análisis de errores
        """
        print(f"\n🔬 Analizando errores de traducción...")
        
        self.model.eval()
        
        length_errors = []  # Errores de longitud
        missing_words = 0
        extra_words = 0
        
        for i, batch in enumerate(tqdm(self.test_loader, desc="Analizando")):
            if i * self.test_loader.batch_size >= max_examples:
                break
            
            src = batch['source_ids'].to(self.device)
            src_mask = batch['source_mask'].to(self.device)
            
            translations = self.translate_batch(src, src_mask)
            
            hyps = self.tgt_tokenizer.batch_decode(
                translations, skip_special_tokens=True
            )
            refs = batch['target_texts']
            
            for hyp, ref in zip(hyps, refs):
                hyp_len = len(hyp.split())
                ref_len = len(ref.split())
                
                length_errors.append(abs(hyp_len - ref_len))
                
                if hyp_len < ref_len:
                    missing_words += ref_len - hyp_len
                elif hyp_len > ref_len:
                    extra_words += hyp_len - ref_len
        
        error_analysis = {
            'avg_length_error': np.mean(length_errors),
            'total_missing_words': missing_words,
            'total_extra_words': extra_words,
            'examples_analyzed': len(length_errors)
        }
        
        print(f"\n📊 Análisis de errores:")
        print(f"   Error promedio de longitud: {error_analysis['avg_length_error']:.2f} palabras")
        print(f"   Palabras faltantes: {missing_words}")
        print(f"   Palabras extra: {extra_words}")
        
        # Guardar análisis
        analysis_file = self.results_dir / "error_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        return error_analysis

def evaluate_all_models(models_dict, test_loader, src_tokenizer, tgt_tokenizer, device=config.DEVICE):
    """
    Evalúa múltiples modelos y compara resultados
    
    Args:
        models_dict: Dict {model_name: model}
        test_loader: DataLoader de test
        src_tokenizer, tgt_tokenizer: Tokenizers
        device: Dispositivo
    
    Returns:
        comparison: DataFrame con comparación de resultados
    """
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Evaluando: {model_name}")
        print(f"{'='*60}")
        
        evaluator = NMTEvaluator(
            model, test_loader, src_tokenizer, tgt_tokenizer,
            model_name, device
        )
        
        bleu_score, result_dict = evaluator.compute_bleu()
        results[model_name] = result_dict
    
    # Guardar comparación
    comparison_file = config.METRICS_DIR / "models_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Comparación guardada en {comparison_file}")
    
    return results

# Ejemplo de uso
if __name__ == "__main__":
    from dataset import create_dataloaders, load_tokenizers
    from model_lstm_attention import create_lstm_model
    from train import load_model_for_inference
    
    # Cargar datos y tokenizers
    _, _, test_loader = create_dataloaders(batch_size=32)
    src_tokenizer, tgt_tokenizer = load_tokenizers()
    
    # Cargar modelo
    model = create_lstm_model(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size
    )
    
    # Cargar checkpoint (asumiendo que existe)
    # model = load_model_for_inference(
    #     model, 
    #     config.MODELS_DIR / "LSTM_Attention" / "best_model.pt"
    # )
    
    # Evaluar
    evaluator = NMTEvaluator(
        model, test_loader, src_tokenizer, tgt_tokenizer, "LSTM_Attention"
    )
    
    # bleu_score, results = evaluator.compute_bleu()
