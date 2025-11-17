"""
Utilidades generales para el proyecto
"""
import sys
import torch
import random
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

def set_seed(seed=config.SEED):
    """
    Establece semilla para reproducibilidad
    
    Args:
        seed: Semilla aleatoria
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Semilla establecida: {seed}")

def count_parameters(model):
    """
    Cuenta parámetros entrenables y totales
    
    Args:
        model: Modelo PyTorch
    
    Returns:
        trainable, total: Número de parámetros
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return trainable, total

def format_time(seconds):
    """
    Formatea segundos en formato legible
    
    Args:
        seconds: Tiempo en segundos
    
    Returns:
        str: Tiempo formateado
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes > 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {secs}s"
    else:
        return f"{minutes}m {secs}s"

def get_device_info():
    """
    Obtiene información del dispositivo
    
    Returns:
        dict: Información del dispositivo
    """
    device = config.DEVICE
    
    info = {
        'device_type': device.type,
        'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['num_gpus'] = torch.cuda.device_count()
        info['memory_allocated'] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        info['memory_reserved'] = f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB"
    
    return info

def print_device_info():
    """Imprime información del dispositivo"""
    info = get_device_info()
    
    print("\n🖥️  INFORMACIÓN DEL DISPOSITIVO")
    print("="*50)
    print(f"Dispositivo: {info['device_type'].upper()}")
    print(f"Nombre: {info['device_name']}")
    
    if info['cuda_available']:
        print(f"CUDA disponible: Sí")
        print(f"Versión CUDA: {info['cuda_version']}")
        print(f"Número de GPUs: {info['num_gpus']}")
        print(f"Memoria asignada: {info['memory_allocated']}")
        print(f"Memoria reservada: {info['memory_reserved']}")
    else:
        print("CUDA disponible: No")
        print("⚠️  Entrenamiento en CPU será LENTO")
    
    print("="*50 + "\n")

def get_model_summary(model, model_name):
    """
    Genera resumen del modelo
    
    Args:
        model: Modelo PyTorch
        model_name: Nombre del modelo
    
    Returns:
        dict: Resumen del modelo
    """
    trainable, total = count_parameters(model)
    
    summary = {
        'model_name': model_name,
        'total_parameters': total,
        'trainable_parameters': trainable,
        'model_size_mb': total * 4 / (1024 ** 2),  # Asumiendo float32
    }
    
    return summary

def print_model_summary(model, model_name):
    """Imprime resumen del modelo"""
    summary = get_model_summary(model, model_name)
    
    print(f"\n📋 RESUMEN DEL MODELO: {model_name}")
    print("="*50)
    print(f"Parámetros totales:     {summary['total_parameters']:,}")
    print(f"Parámetros entrenables: {summary['trainable_parameters']:,}")
    print(f"Tamaño estimado:        {summary['model_size_mb']:.2f} MB")
    print("="*50 + "\n")

def save_model_architecture(model, model_name, filepath=None):
    """
    Guarda la arquitectura del modelo en archivo de texto
    
    Args:
        model: Modelo PyTorch
        model_name: Nombre del modelo
        filepath: Ruta del archivo (opcional)
    """
    if filepath is None:
        filepath = config.MODELS_DIR / model_name / "architecture.txt"
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(f"ARQUITECTURA DEL MODELO: {model_name}\n")
        f.write("="*70 + "\n\n")
        f.write(str(model))
        f.write("\n\n" + "="*70 + "\n")
        
        summary = get_model_summary(model, model_name)
        f.write(f"\nPARÁMETROS TOTALES: {summary['total_parameters']:,}\n")
        f.write(f"PARÁMETROS ENTRENABLES: {summary['trainable_parameters']:,}\n")
        f.write(f"TAMAÑO ESTIMADO: {summary['model_size_mb']:.2f} MB\n")
    
    print(f"✅ Arquitectura guardada en: {filepath}")

def check_gpu_memory():
    """Verifica memoria GPU disponible"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memoria total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Memoria asignada: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memoria en caché: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"  Memoria libre: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1e9:.2f} GB")
    else:
        print("❌ No hay GPU disponible")

def clear_gpu_memory():
    """Limpia memoria GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ Memoria GPU limpiada")

# Ejecutar al importar
set_seed(config.SEED)

if __name__ == "__main__":
    print_device_info()
    check_gpu_memory()
