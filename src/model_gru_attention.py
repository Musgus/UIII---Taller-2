"""
MODELO 3: GRU con Atención Bahdanau
Similar al LSTM pero usando GRU (Gated Recurrent Unit)
GRU es más simple que LSTM (no tiene cell state separado)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# Reutilizamos la atención Bahdanau del modelo LSTM
from model_lstm_attention import BahdanauAttention

class GRUEncoder(nn.Module):
    """
    Encoder con GRU bidireccional
    GRU es más simple que LSTM: solo tiene hidden state (no cell state)
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3, bidirectional=True):
        super(GRUEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=config.PAD_IDX
        )
        
        # GRU - Principal diferencia con LSTM: solo maneja hidden, no cell
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Si es bidireccional, proyectar a hidden_dim
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src, src_lengths=None):
        """
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
        
        Returns:
            outputs: (batch, src_len, hidden_dim)
            hidden: (num_layers, batch, hidden_dim)
        """
        # Embedding
        embedded = self.dropout(self.embedding(src))
        
        # GRU - retorna solo outputs y hidden (no cell como LSTM)
        outputs, hidden = self.gru(embedded)
        
        # Si es bidireccional, combinar direcciones
        if self.bidirectional:
            # Proyectar outputs
            outputs = self.fc_hidden(outputs)
            
            # Combinar forward y backward hidden states
            # hidden: (num_layers*2, batch, hidden) -> (num_layers, batch, hidden*2)
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
            hidden = self.fc_hidden(hidden)
        
        return outputs, hidden

class GRUAttentionDecoder(nn.Module):
    """
    Decoder con GRU y mecanismo de atención Bahdanau
    Similar al LSTM decoder pero más simple (sin cell state)
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(GRUAttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=config.PAD_IDX
        )
        
        # Atención (misma que LSTM)
        self.attention = BahdanauAttention(hidden_dim)
        
        # GRU - input = embedding + context
        self.gru = nn.GRU(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Capa de salida
        self.fc_out = nn.Linear(
            hidden_dim + hidden_dim + embedding_dim,
            vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, hidden, encoder_outputs, src_mask=None):
        """
        Args:
            tgt: (batch, 1) - Token actual
            hidden: (num_layers, batch, hidden_dim) - Solo hidden, no cell
            encoder_outputs: (batch, src_len, hidden_dim)
            src_mask: (batch, src_len)
        
        Returns:
            output: (batch, vocab_size)
            hidden: (num_layers, batch, hidden_dim)
            attention_weights: (batch, src_len)
        """
        # Embedding
        embedded = self.dropout(self.embedding(tgt))
        
        # Calcular atención
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs, src_mask
        )
        
        # Concatenar embedding con context
        context = context.unsqueeze(1)
        gru_input = torch.cat([embedded, context], dim=2)
        
        # GRU - más simple que LSTM, solo retorna output y hidden
        output, hidden = self.gru(gru_input, hidden)
        
        # Preparar output final
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        context = context.squeeze(1)
        
        output = self.fc_out(
            torch.cat([output, context, embedded], dim=1)
        )
        
        return output, hidden, attention_weights

class Seq2SeqGRUAttention(nn.Module):
    """
    Modelo completo Seq2Seq con GRU y Atención Bahdanau
    
    Ventajas de GRU vs LSTM:
    - Menos parámetros (no tiene cell state)
    - Más rápido de entrenar
    - Similar rendimiento en muchos casos
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, config_dict):
        super(Seq2SeqGRUAttention, self).__init__()
        
        embedding_dim = config_dict['embedding_dim']
        hidden_dim = config_dict['hidden_dim']
        num_layers = config_dict['num_layers']
        dropout = config_dict['dropout']
        bidirectional = config_dict.get('bidirectional', True)
        
        self.encoder = GRUEncoder(
            src_vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout,
            bidirectional
        )
        
        self.decoder = GRUAttentionDecoder(
            tgt_vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout
        )
        
        self.tgt_vocab_size = tgt_vocab_size
    
    def forward(self, src, tgt, src_mask=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
            src_mask: (batch_size, src_len)
            teacher_forcing_ratio: Probabilidad de usar ground truth
        
        Returns:
            outputs: (batch_size, tgt_len, vocab_size)
            attention_weights: Lista de pesos de atención por paso
        """
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        
        # Encoder - GRU retorna solo outputs y hidden (no cell)
        encoder_outputs, hidden = self.encoder(src)
        
        # Preparar outputs
        outputs = torch.zeros(
            batch_size, tgt_len, self.tgt_vocab_size
        ).to(tgt.device)
        
        attention_weights_all = []
        
        # Primer input del decoder
        decoder_input = tgt[:, 0].unsqueeze(1)
        
        # Decodificación paso a paso
        for t in range(1, tgt_len):
            # GRU decoder: no necesita cell state
            output, hidden, attn_weights = self.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )
            
            outputs[:, t] = output
            attention_weights_all.append(attn_weights)
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs, attention_weights_all
    
    def generate(self, src, src_mask=None, max_length=config.MAX_LENGTH, device=config.DEVICE):
        """
        Generación greedy para inferencia
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Encoder
            encoder_outputs, hidden = self.encoder(src)
            
            # Inicializar con <bos>
            decoder_input = torch.full(
                (batch_size, 1), config.BOS_IDX, dtype=torch.long
            ).to(device)
            
            outputs = [decoder_input]
            attention_weights_all = []
            
            # Generar token por token
            for _ in range(max_length - 1):
                output, hidden, attn_weights = self.decoder(
                    decoder_input, hidden, encoder_outputs, src_mask
                )
                
                top1 = output.argmax(1).unsqueeze(1)
                outputs.append(top1)
                attention_weights_all.append(attn_weights)
                
                decoder_input = top1
                
                # Parar si todos generaron <eos>
                if (top1 == config.EOS_IDX).all():
                    break
            
            return torch.cat(outputs, dim=1), attention_weights_all

def create_gru_model(src_vocab_size, tgt_vocab_size):
    """Factory function para crear el modelo GRU con atención"""
    model = Seq2SeqGRUAttention(src_vocab_size, tgt_vocab_size, config.GRU_CONFIG)
    return model

# Test del modelo
if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 4
    src_len = 15
    tgt_len = 12
    
    model = create_gru_model(src_vocab_size, tgt_vocab_size)
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    src_mask = (src == config.PAD_IDX)
    
    outputs, attn_weights = model(src, tgt, src_mask, teacher_forcing_ratio=0.5)
    
    print("✅ Modelo GRU con Atención Bahdanau:")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Attention weights: {len(attn_weights)} pasos")
    print(f"   Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Comparación de parámetros GRU vs LSTM
    print("\n📊 GRU tiene ~25% menos parámetros que LSTM equivalente")
    print("   (no tiene cell state separado)")
