"""
Sistema de entrenamiento completo con:
- Checkpointing automático
- Early stopping
- Logging detallado
- Guardado de métricas
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import json
from pathlib import Path
from tqdm import tqdm

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

class Trainer:
    """
    Entrenador genérico para todos los modelos NMT
    """
    
    def __init__(self, model, train_loader, valid_loader, model_name,
                 learning_rate=config.LEARNING_RATE,
                 device=config.DEVICE):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_name = model_name
        self.device = device
        
        # Criterio de loss (ignorar padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
        
        # Optimizador
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Scheduler: reduce LR si valid loss no mejora
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        # Directorios de guardado
        self.model_dir = config.MODELS_DIR / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = config.METRICS_DIR / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Historial de entrenamiento
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Para early stopping
        self.best_valid_loss = float('inf')
        self.patience_counter = 0
        
        # Contar parámetros
        self.num_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n🎯 Trainer inicializado para {model_name}")
        print(f"   Parámetros del modelo: {self.num_params:,}")
        print(f"   Dispositivo: {device}")
        print(f"   Learning rate: {learning_rate}")
    
    def train_epoch(self, teacher_forcing_ratio=0.5):
        """Entrena una época completa"""
        self.model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover datos al device
            src = batch['source_ids'].to(self.device)
            tgt = batch['target_ids'].to(self.device)
            src_mask = batch['source_mask'].to(self.device)
            tgt_mask = batch.get('target_mask', None)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Diferentes modelos tienen diferentes firmas forward
            if 'Transformer' in self.model_name:
                output = self.model(src, tgt, src_mask, tgt_mask)
            elif 'RNN_Simple' in self.model_name:
                output = self.model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
            else:  # LSTM o GRU con atención
                output, _ = self.model(src, tgt, src_mask, teacher_forcing_ratio)
            
            # Calcular loss
            # output: (batch, tgt_len, vocab_size)
            # tgt: (batch, tgt_len)
            output = output[:, 1:].reshape(-1, output.shape[-1])  # Skip <bos>
            tgt = tgt[:, 1:].reshape(-1)  # Skip <bos>
            
            loss = self.criterion(output, tgt)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Actualizar progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """Valida el modelo"""
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="Validating"):
                src = batch['source_ids'].to(self.device)
                tgt = batch['target_ids'].to(self.device)
                src_mask = batch['source_mask'].to(self.device)
                tgt_mask = batch.get('target_mask', None)
                if tgt_mask is not None:
                    tgt_mask = tgt_mask.to(self.device)
                
                # Forward pass (sin teacher forcing en validación)
                if 'Transformer' in self.model_name:
                    output = self.model(src, tgt, src_mask, tgt_mask)
                elif 'RNN_Simple' in self.model_name:
                    output = self.model(src, tgt, teacher_forcing_ratio=0)
                else:
                    output, _ = self.model(src, tgt, src_mask, teacher_forcing_ratio=0)
                
                # Calcular loss
                output = output[:, 1:].reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = self.criterion(output, tgt)
                epoch_loss += loss.item()
        
        return epoch_loss / len(self.valid_loader)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Guarda checkpoint del modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_valid_loss': self.best_valid_loss,
            'num_params': self.num_params
        }
        
        # Guardar checkpoint regular
        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar último checkpoint
        last_checkpoint_path = self.model_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_checkpoint_path)
        
        # Si es el mejor, guardarlo aparte
        if is_best:
            best_checkpoint_path = self.model_dir / "best_model.pt"
            torch.save(checkpoint, best_checkpoint_path)
            print(f"   💾 Mejor modelo guardado (valid_loss: {self.best_valid_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Carga checkpoint para continuar entrenamiento"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_valid_loss = checkpoint['best_valid_loss']
        
        print(f"✅ Checkpoint cargado desde época {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def save_metrics(self):
        """Guarda métricas de entrenamiento en JSON"""
        metrics = {
            'model_name': self.model_name,
            'num_params': self.num_params,
            'history': self.history,
            'best_valid_loss': self.best_valid_loss,
            'total_epochs': len(self.history['train_loss']),
            'total_training_time': sum(self.history['epoch_times']),
        }
        
        metrics_file = self.results_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   📊 Métricas guardadas en {metrics_file}")
    
    def train(self, num_epochs=config.NUM_EPOCHS, 
              early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
              teacher_forcing_ratio=0.5):
        """
        Bucle de entrenamiento completo
        
        Args:
            num_epochs: Número de épocas
            early_stopping_patience: Paciencia para early stopping
            teacher_forcing_ratio: Ratio de teacher forcing (decrece con épocas)
        """
        print(f"\n{'='*60}")
        print(f"🚀 ENTRENAMIENTO DE {self.model_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            print(f"\n📅 Época {epoch}/{num_epochs}")
            print("-" * 40)
            
            # Decaimiento de teacher forcing
            current_tf_ratio = teacher_forcing_ratio * (0.95 ** (epoch - 1))
            
            # Entrenar
            train_loss = self.train_epoch(teacher_forcing_ratio=current_tf_ratio)
            
            # Validar
            valid_loss = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Guardar en historial
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Actualizar scheduler
            self.scheduler.step(valid_loss)
            
            # Mostrar resultados
            print(f"\n📈 Resultados de la época:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Valid Loss: {valid_loss:.4f}")
            print(f"   Tiempo: {epoch_time:.2f}s")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"   Teacher Forcing: {current_tf_ratio:.4f}")
            
            # Guardar checkpoint
            is_best = valid_loss < self.best_valid_loss
            
            if is_best:
                self.best_valid_loss = valid_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Guardar checkpoint cada época o si es el mejor
            if epoch % config.SAVE_CHECKPOINT_EVERY == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping activado (paciencia: {early_stopping_patience})")
                break
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"⏱️  Tiempo total: {total_time/60:.2f} minutos")
        print(f"🏆 Mejor valid loss: {self.best_valid_loss:.4f}")
        print(f"📊 Épocas entrenadas: {len(self.history['train_loss'])}")
        
        # Guardar métricas finales
        self.save_metrics()
        
        return self.history

def load_model_for_inference(model, checkpoint_path, device=config.DEVICE):
    """
    Carga un modelo para inferencia desde un checkpoint
    
    Args:
        model: Instancia del modelo (arquitectura)
        checkpoint_path: Ruta al checkpoint (.pt)
        device: Dispositivo
    
    Returns:
        model: Modelo cargado en modo eval
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado para inferencia desde {checkpoint_path}")
    print(f"   Época: {checkpoint['epoch']}")
    print(f"   Mejor valid loss: {checkpoint['best_valid_loss']:.4f}")
    
    return model

# Ejemplo de uso
if __name__ == "__main__":
    from dataset import create_dataloaders
    from model_rnn import create_rnn_model
    
    # Cargar datos
    train_loader, valid_loader, test_loader = create_dataloaders(batch_size=32)
    
    # Crear modelo
    model = create_rnn_model(src_vocab_size=16000, tgt_vocab_size=16000)
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model_name="RNN_Simple_test"
    )
    
    # Entrenar (pocas épocas para test)
    trainer.train(num_epochs=2)
