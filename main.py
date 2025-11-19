"""
Script principal para entrenar y evaluar todos los modelos NMT
Pipeline completo: preparación → entrenamiento → evaluación → visualización
"""
import torch
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from src.dataset import create_dataloaders, load_tokenizers
from src.model_rnn import create_rnn_model
from src.model_lstm_attention import create_lstm_model
from src.model_gru_attention import create_gru_model
from src.model_transformer import create_transformer_model
from src.train import Trainer, load_model_for_inference
from src.evaluate import NMTEvaluator, evaluate_all_models
from src.visualize import ResultsVisualizer

def check_data_ready():
    """Verifica que los datos estén preparados"""
    required_files = [
        config.PROCESSED_DATA_DIR / "train.jsonl",
        config.PROCESSED_DATA_DIR / "valid.jsonl",
        config.PROCESSED_DATA_DIR / "test.jsonl",
        config.TOKENIZER_DIR / f"spm_{config.SOURCE_LANG}.model",
        config.TOKENIZER_DIR / f"spm_{config.TARGET_LANG}.model",
    ]
    
    all_ready = all(f.exists() for f in required_files)
    
    if not all_ready:
        print("❌ Datos no preparados. Ejecuta primero:")
        print("   1. python src/download_data.py")
        print("   2. python src/preprocess.py")
        return False
    
    return True

def train_all_models(train_loader, valid_loader, src_vocab_size, tgt_vocab_size):
    """
    Entrena los 4 modelos
    
    Returns:
        models_dict: Diccionario con los modelos entrenados
    """
    print(f"\n{'='*70}")
    print("🚀 FASE DE ENTRENAMIENTO - TODOS LOS MODELOS")
    print(f"{'='*70}\n")
    
    models_dict = {}
    
    # Configuraciones de modelos
    model_configs = [
        ("RNN_Simple", create_rnn_model, config.RNN_CONFIG),
        ("LSTM_Attention", create_lstm_model, config.LSTM_CONFIG),
        ("GRU_Attention", create_gru_model, config.GRU_CONFIG),
        ("Transformer", create_transformer_model, config.TRANSFORMER_CONFIG),
    ]
    
    for model_name, create_fn, model_config in model_configs:
        print(f"\n{'#'*70}")
        print(f"# ENTRENANDO: {model_name}")
        print(f"{'#'*70}")
        
        # Crear modelo
        model = create_fn(src_vocab_size, tgt_vocab_size)
        
        # Crear trainer
        learning_rate = model_config.get('learning_rate', config.LEARNING_RATE)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            model_name=model_name,
            learning_rate=learning_rate
        )
        
        # Entrenar
        try:
            trainer.train(
                num_epochs=config.NUM_EPOCHS,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                teacher_forcing_ratio=0.5
            )
            
            # Cargar mejor modelo
            best_model_path = config.MODELS_DIR / model_name / "best_model.pt"
            if best_model_path.exists():
                model = load_model_for_inference(model, best_model_path)
                models_dict[model_name] = model
            
        except Exception as e:
            print(f"❌ Error entrenando {model_name}: {e}")
            continue
    
    return models_dict

def evaluate_all_models_pipeline(models_dict, test_loader, src_tokenizer, tgt_tokenizer):
    """
    Evalúa todos los modelos
    
    Args:
        models_dict: Diccionario con modelos entrenados
        test_loader: DataLoader de test
        src_tokenizer, tgt_tokenizer: Tokenizers
    
    Returns:
        results: Diccionario con resultados de evaluación
    """
    print(f"\n{'='*70}")
    print("📊 FASE DE EVALUACIÓN - TODOS LOS MODELOS")
    print(f"{'='*70}\n")
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'#'*70}")
        print(f"# EVALUANDO: {model_name}")
        print(f"{'#'*70}")
        
        try:
            evaluator = NMTEvaluator(
                model=model,
                test_loader=test_loader,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                model_name=model_name
            )
            
            bleu_score, result_dict = evaluator.compute_bleu(save_translations=True)
            evaluator.analyze_errors(max_examples=200)
            
            results[model_name] = result_dict
            
        except Exception as e:
            print(f"❌ Error evaluando {model_name}: {e}")
            continue
    
    return results

def visualize_all_results(model_names):
    """
    Genera todas las visualizaciones
    
    Args:
        model_names: Lista de nombres de modelos
    """
    print(f"\n{'='*70}")
    print("🎨 FASE DE VISUALIZACIÓN")
    print(f"{'='*70}\n")
    
    visualizer = ResultsVisualizer()
    visualizer.generate_all_plots(model_names)

def print_final_summary(results):
    """
    Imprime un resumen final de todos los resultados
    
    Args:
        results: Diccionario con resultados de evaluación
    """
    print(f"\n{'='*70}")
    print("📈 RESUMEN FINAL DE RESULTADOS")
    print(f"{'='*70}\n")
    
    # Encontrar mejor modelo por BLEU
    best_model = max(results.items(), key=lambda x: x[1]['bleu_score'])
    
    print("🏆 RANKING POR BLEU SCORE:")
    print("-" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['bleu_score'], reverse=True)
    
    for i, (model_name, result) in enumerate(sorted_results, 1):
        bleu = result['bleu_score']
        print(f"   {i}. {model_name:20s} BLEU: {bleu:6.2f}")
    
    print(f"\n🥇 MEJOR MODELO: {best_model[0]} (BLEU: {best_model[1]['bleu_score']:.2f})")
    
    # Cargar métricas de entrenamiento del mejor modelo
    import json
    metrics_file = config.METRICS_DIR / best_model[0] / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"\n📊 DETALLES DEL MEJOR MODELO:")
        print(f"   Parámetros:      {metrics['num_params']:,}")
        print(f"   Épocas:          {metrics['total_epochs']}")
        print(f"   Tiempo:          {metrics['total_training_time']/60:.2f} minutos")
        print(f"   Valid Loss:      {metrics['best_valid_loss']:.4f}")
    
    print("\n" + "="*70)
    print(f"✅ Pipeline completo finalizado")
    print(f"📁 Resultados en: {config.RESULTADOS_DIR}")
    print(f"📁 Modelos en: {config.MODELS_DIR}")
    print("="*70 + "\n")

def main(skip_training=False, skip_evaluation=False):
    """
    Pipeline completo
    
    Args:
        skip_training: Si True, carga modelos existentes en lugar de entrenar
        skip_evaluation: Si True, salta evaluación y solo visualiza
    """
    print("\n" + "="*70)
    print("🌟 PROYECTO NMT - TRADUCCIÓN AUTOMÁTICA NEURONAL")
    print(f"   Par de idiomas: {config.SOURCE_LANG} → {config.TARGET_LANG}")
    print("="*70)
    
    # 1. Verificar datos
    if not check_data_ready():
        return
    
    print("\n✅ Datos preparados y listos")
    
    # 2. Cargar dataloaders y tokenizers
    print("\n📚 Cargando datos y tokenizers...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE
    )
    src_tokenizer, tgt_tokenizer = load_tokenizers()
    
    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size
    
    print(f"   Vocab size (source): {src_vocab_size:,}")
    print(f"   Vocab size (target): {tgt_vocab_size:,}")
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Valid batches: {len(valid_loader):,}")
    print(f"   Test batches: {len(test_loader):,}")
    
    models_dict = {}
    model_names = ["RNN_Simple", "LSTM_Attention", "GRU_Attention", "Transformer"]
    
    # 3. Entrenamiento
    if not skip_training:
        models_dict = train_all_models(
            train_loader, valid_loader, src_vocab_size, tgt_vocab_size
        )
    else:
        print("\n⏭️  Saltando entrenamiento, cargando modelos existentes...")
        
        # Cargar modelos desde checkpoints
        model_creators = {
            "RNN_Simple": create_rnn_model,
            "LSTM_Attention": create_lstm_model,
            "GRU_Attention": create_gru_model,
            "Transformer": create_transformer_model,
        }
        
        for model_name, create_fn in model_creators.items():
            checkpoint_path = config.MODELS_DIR / model_name / "best_model.pt"
            
            if checkpoint_path.exists():
                model = create_fn(src_vocab_size, tgt_vocab_size)
                model = load_model_for_inference(model, checkpoint_path)
                models_dict[model_name] = model
            else:
                print(f"   ⚠️  Checkpoint no encontrado para {model_name}")
    
    # 4. Evaluación
    results = {}
    if not skip_evaluation and models_dict:
        results = evaluate_all_models_pipeline(
            models_dict, test_loader, src_tokenizer, tgt_tokenizer
        )
    
    # 5. Visualización
    # Filtrar solo modelos que tienen resultados
    available_models = []
    for model_name in model_names:
        metrics_file = config.METRICS_DIR / model_name / "training_metrics.json"
        if metrics_file.exists():
            available_models.append(model_name)
    
    if available_models:
        visualize_all_results(available_models)
    
    # 6. Resumen final
    if results:
        print_final_summary(results)
    
    print("\n🎉 ¡Pipeline completado exitosamente!\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline completo NMT")
    parser.add_argument('--skip-training', action='store_true',
                       help='Saltar entrenamiento y cargar modelos existentes')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Saltar evaluación (solo visualizar)')
    
    args = parser.parse_args()
    
    main(
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )
