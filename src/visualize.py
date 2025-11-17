"""
Visualización de resultados:
- Curvas de loss
- Comparación de BLEU
- Análisis de atención
- Gráficos comparativos
"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class ResultsVisualizer:
    """
    Visualizador de resultados de entrenamiento y evaluación
    """
    
    def __init__(self):
        self.plots_dir = config.PLOTS_DIR
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Visualizador inicializado")
        print(f"   Gráficos se guardarán en: {self.plots_dir}")
    
    def plot_training_curves(self, model_name):
        """
        Grafica curvas de loss de entrenamiento y validación
        
        Args:
            model_name: Nombre del modelo
        """
        # Cargar métricas
        metrics_file = config.METRICS_DIR / model_name / "training_metrics.json"
        
        if not metrics_file.exists():
            print(f"❌ No se encontraron métricas para {model_name}")
            return
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        history = metrics['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['valid_loss'], 'r-', label='Valid Loss', linewidth=2)
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{model_name} - Curvas de Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title(f'{model_name} - Learning Rate', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        output_file = self.plots_dir / f"{model_name}_training_curves.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Gráfico guardado: {output_file}")
    
    def plot_all_training_curves(self, model_names):
        """
        Compara curvas de loss de múltiples modelos
        
        Args:
            model_names: Lista de nombres de modelos
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = ['b', 'r', 'g', 'purple']
        
        for i, model_name in enumerate(model_names):
            metrics_file = config.METRICS_DIR / model_name / "training_metrics.json"
            
            if not metrics_file.exists():
                continue
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            history = metrics['history']
            epochs = range(1, len(history['train_loss']) + 1)
            
            color = colors[i % len(colors)]
            
            # Train loss
            ax1.plot(epochs, history['train_loss'], 
                    color=color, linestyle='-', label=model_name, linewidth=2)
            
            # Valid loss
            ax2.plot(epochs, history['valid_loss'],
                    color=color, linestyle='-', label=model_name, linewidth=2)
        
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Train Loss', fontsize=12)
        ax1.set_title('Comparación - Train Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('Valid Loss', fontsize=12)
        ax2.set_title('Comparación - Valid Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.plots_dir / "all_models_training_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comparación guardada: {output_file}")
    
    def plot_bleu_comparison(self, model_names):
        """
        Gráfico de barras comparando BLEU scores
        
        Args:
            model_names: Lista de nombres de modelos
        """
        bleu_scores = []
        valid_losses = []
        names = []
        
        for model_name in model_names:
            eval_file = config.METRICS_DIR / model_name / "evaluation_results.json"
            
            if not eval_file.exists():
                continue
            
            with open(eval_file, 'r') as f:
                results = json.load(f)
            
            bleu_scores.append(results['bleu_score'])
            names.append(model_name)
            
            # Obtener valid loss
            metrics_file = config.METRICS_DIR / model_name / "training_metrics.json"
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            valid_losses.append(metrics['best_valid_loss'])
        
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # BLEU scores
        bars1 = ax1.bar(names, bleu_scores, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
        ax1.set_ylabel('BLEU Score', fontsize=12)
        ax1.set_title('Comparación de BLEU Scores', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(bleu_scores) * 1.2)
        
        # Agregar valores en las barras
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # Valid Loss
        bars2 = ax2.bar(names, valid_losses, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Comparación de Validation Loss', fontsize=14, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        plt.tight_layout()
        
        output_file = self.plots_dir / "bleu_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comparación BLEU guardada: {output_file}")
    
    def plot_bleu_by_length(self, model_names):
        """
        Compara BLEU por longitud de oración
        
        Args:
            model_names: Lista de nombres de modelos
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        length_categories = ['short', 'medium', 'long']
        x = np.arange(len(length_categories))
        width = 0.2
        
        for i, model_name in enumerate(model_names):
            eval_file = config.METRICS_DIR / model_name / "evaluation_results.json"
            
            if not eval_file.exists():
                continue
            
            with open(eval_file, 'r') as f:
                results = json.load(f)
            
            bleu_by_length = results.get('bleu_by_length', {})
            scores = [bleu_by_length.get(cat, 0) for cat in length_categories]
            
            offset = width * (i - len(model_names)/2 + 0.5)
            bars = ax.bar(x + offset, scores, width, label=model_name)
            
            # Agregar valores
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Longitud de Oración', fontsize=12)
        ax.set_ylabel('BLEU Score', fontsize=12)
        ax.set_title('BLEU Score por Longitud de Oración', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Corta\n(≤10 palabras)', 'Media\n(11-20)', 'Larga\n(>20)'])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.plots_dir / "bleu_by_length.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ BLEU por longitud guardado: {output_file}")
    
    def plot_model_comparison_table(self, model_names):
        """
        Crea una tabla comparativa de todos los modelos
        
        Args:
            model_names: Lista de nombres de modelos
        """
        data = []
        
        for model_name in model_names:
            row = {'Modelo': model_name}
            
            # Métricas de entrenamiento
            metrics_file = config.METRICS_DIR / model_name / "training_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                row['Parámetros'] = f"{metrics['num_params']:,}"
                row['Épocas'] = metrics['total_epochs']
                row['Tiempo (min)'] = f"{metrics['total_training_time']/60:.1f}"
                row['Valid Loss'] = f"{metrics['best_valid_loss']:.4f}"
            
            # Métricas de evaluación
            eval_file = config.METRICS_DIR / model_name / "evaluation_results.json"
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    results = json.load(f)
                
                row['BLEU'] = f"{results['bleu_score']:.2f}"
            
            data.append(row)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(14, len(data) * 0.8 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Crear tabla
        table_data = []
        headers = ['Modelo', 'Parámetros', 'Épocas', 'Tiempo (min)', 'Valid Loss', 'BLEU']
        
        for row in data:
            table_data.append([row.get(h, 'N/A') for h in headers])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.15, 0.1, 0.15, 0.15, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Estilo
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        plt.title('Comparación Completa de Modelos NMT', 
                 fontsize=16, fontweight='bold', pad=20)
        
        output_file = self.plots_dir / "models_comparison_table.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Tabla comparativa guardada: {output_file}")
    
    def plot_training_time_comparison(self, model_names):
        """
        Compara tiempos de entrenamiento
        
        Args:
            model_names: Lista de nombres de modelos
        """
        times = []
        names = []
        
        for model_name in model_names:
            metrics_file = config.METRICS_DIR / model_name / "training_metrics.json"
            
            if not metrics_file.exists():
                continue
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            times.append(metrics['total_training_time'] / 60)  # En minutos
            names.append(model_name)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(names, times, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
        ax.set_xlabel('Tiempo de Entrenamiento (minutos)', fontsize=12)
        ax.set_title('Comparación de Tiempo de Entrenamiento', fontsize=14, fontweight='bold')
        
        # Agregar valores
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(time, i, f' {time:.1f} min',
                   va='center', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        output_file = self.plots_dir / "training_time_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comparación de tiempos guardada: {output_file}")
    
    def generate_all_plots(self, model_names):
        """
        Genera todos los gráficos de análisis
        
        Args:
            model_names: Lista de nombres de modelos
        """
        print(f"\n{'='*60}")
        print("🎨 GENERANDO TODOS LOS GRÁFICOS")
        print(f"{'='*60}\n")
        
        # Curvas individuales
        for model_name in model_names:
            self.plot_training_curves(model_name)
        
        # Comparaciones
        self.plot_all_training_curves(model_names)
        self.plot_bleu_comparison(model_names)
        self.plot_bleu_by_length(model_names)
        self.plot_model_comparison_table(model_names)
        self.plot_training_time_comparison(model_names)
        
        print(f"\n✅ Todos los gráficos generados en: {self.plots_dir}")

# Ejemplo de uso
if __name__ == "__main__":
    visualizer = ResultsVisualizer()
    
    model_names = [
        "RNN_Simple",
        "LSTM_Attention",
        "GRU_Attention",
        "Transformer"
    ]
    
    # visualizer.generate_all_plots(model_names)
