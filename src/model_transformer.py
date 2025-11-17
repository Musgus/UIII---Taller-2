"""
MODELO 4: Transformer Simplificado
Implementación de arquitectura Transformer para traducción
Basado en "Attention is All You Need" (Vaswani et al., 2017)
Versión simplificada con menos capas para entrenamiento eficiente
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal
    Agrega información de posición a los embeddings
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Crear matriz de codificación posicional
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x con codificación posicional agregada
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """
    Encoder del Transformer
    Múltiples capas de self-attention + feed-forward
    """
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Capas de encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Escalar embeddings
        self.scale = math.sqrt(d_model)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (batch, src_len) - IDs de tokens
            src_mask: (src_len, src_len) - Máscara de atención
            src_key_padding_mask: (batch, src_len) - Máscara de padding
        
        Returns:
            memory: (batch, src_len, d_model) - Salidas del encoder
        """
        # Embedding + scaling + positional encoding
        src = self.embedding(src) * self.scale  # (batch, src_len, d_model)
        src = self.pos_encoder(src)
        
        # Encoder
        memory = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return memory

class TransformerDecoder(nn.Module):
    """
    Decoder del Transformer
    Múltiples capas de self-attention + cross-attention + feed-forward
    """
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Capas de decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Capa de salida
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Escalar embeddings
        self.scale = math.sqrt(d_model)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: (batch, tgt_len) - Secuencia target
            memory: (batch, src_len, d_model) - Salida del encoder
            tgt_mask: (tgt_len, tgt_len) - Máscara causal
            memory_mask: (tgt_len, src_len) - Máscara de memoria
            tgt_key_padding_mask: (batch, tgt_len) - Padding del target
            memory_key_padding_mask: (batch, src_len) - Padding del source
        
        Returns:
            output: (batch, tgt_len, vocab_size)
        """
        # Embedding + scaling + positional encoding
        tgt = self.embedding(tgt) * self.scale
        tgt = self.pos_encoder(tgt)
        
        # Decoder
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Proyectar a vocabulario
        output = self.fc_out(output)
        
        return output

class Seq2SeqTransformer(nn.Module):
    """
    Modelo completo Transformer para traducción
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, config_dict):
        super(Seq2SeqTransformer, self).__init__()
        
        d_model = config_dict['d_model']
        nhead = config_dict['nhead']
        num_encoder_layers = config_dict['num_encoder_layers']
        num_decoder_layers = config_dict['num_decoder_layers']
        dim_feedforward = config_dict['dim_feedforward']
        dropout = config_dict['dropout']
        
        self.encoder = TransformerEncoder(
            src_vocab_size,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            d_model,
            nhead,
            num_decoder_layers,
            dim_feedforward,
            dropout
        )
        
        self.tgt_vocab_size = tgt_vocab_size
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización Xavier para mejor convergencia"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """
        Genera máscara causal (triangular superior)
        Previene que el decoder vea tokens futuros
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
            src_mask: (batch, src_len) - True donde hay padding
            tgt_mask: (batch, tgt_len) - True donde hay padding
        
        Returns:
            output: (batch, tgt_len, vocab_size)
        """
        # Crear máscaras
        device = src.device
        tgt_len = tgt.shape[1]
        
        # Máscara causal para el decoder
        tgt_causal_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
        
        # Encoder
        memory = self.encoder(
            src,
            src_key_padding_mask=src_mask
        )
        
        # Decoder
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        
        return output
    
    def generate(self, src, src_mask=None, max_length=config.MAX_LENGTH, device=config.DEVICE):
        """
        Generación autoregresiva greedy
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Encode una sola vez
            memory = self.encoder(
                src,
                src_key_padding_mask=src_mask
            )
            
            # Inicializar con <bos>
            ys = torch.full((batch_size, 1), config.BOS_IDX, dtype=torch.long).to(device)
            
            for i in range(max_length - 1):
                # Crear máscara causal
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)
                
                # Decode
                out = self.decoder(
                    ys,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_mask
                )
                
                # Siguiente token (greedy)
                next_token = out[:, -1].argmax(dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_token], dim=1)
                
                # Parar si todos generaron <eos>
                if (next_token == config.EOS_IDX).all():
                    break
            
            return ys

def create_transformer_model(src_vocab_size, tgt_vocab_size):
    """Factory function para crear el modelo Transformer"""
    model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, config.TRANSFORMER_CONFIG)
    return model

# Test del modelo
if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 4
    src_len = 15
    tgt_len = 12
    
    model = create_transformer_model(src_vocab_size, tgt_vocab_size)
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    src_mask = (src == config.PAD_IDX)
    tgt_mask = (tgt == config.PAD_IDX)
    
    outputs = model(src, tgt, src_mask, tgt_mask)
    
    print("✅ Modelo Transformer:")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test de generación
    generated = model.generate(src, src_mask, max_length=20)
    print(f"   Generated shape: {generated.shape}")
    
    print("\n💡 Ventajas del Transformer:")
    print("   - Paralelizable (no secuencial como RNN/LSTM/GRU)")
    print("   - Captura dependencias a largo plazo mejor")
    print("   - Estado del arte en NMT")
