"""
MODELO 1: RNN Simple sin Atención
Arquitectura encoder-decoder básica con SimpleRNN
"""
import torch
import torch.nn as nn
import config

class RNNEncoder(nn.Module):
    """
    Encoder con SimpleRNN
    Procesa la secuencia de entrada y produce un vector de contexto
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(RNNEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Capa de embedding
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=config.PAD_IDX
        )
        
        # SimpleRNN (vanilla RNN)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_lengths=None):
        """
        Args:
            src: (batch_size, src_len) - IDs de tokens
            src_lengths: (batch_size,) - longitudes reales
        
        Returns:
            outputs: (batch_size, src_len, hidden_dim)
            hidden: (num_layers, batch_size, hidden_dim)
        """
        # Embedding
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb_dim)
        
        # RNN
        # outputs: (batch, src_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden

class RNNDecoder(nn.Module):
    """
    Decoder con SimpleRNN (sin atención)
    Usa el último estado del encoder como contexto inicial
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(RNNDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=config.PAD_IDX
        )
        
        # RNN
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Capa de salida
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, hidden):
        """
        Args:
            tgt: (batch_size, 1) - Token actual (un paso a la vez)
            hidden: (num_layers, batch_size, hidden_dim) - Estado anterior
        
        Returns:
            output: (batch_size, vocab_size) - Probabilidades del siguiente token
            hidden: (num_layers, batch_size, hidden_dim) - Nuevo estado
        """
        # Embedding
        embedded = self.dropout(self.embedding(tgt))  # (batch, 1, emb_dim)
        
        # RNN (un paso)
        output, hidden = self.rnn(embedded, hidden)  # output: (batch, 1, hidden_dim)
        
        # Proyectar a vocabulario
        output = self.fc_out(output.squeeze(1))  # (batch, vocab_size)
        
        return output, hidden

class Seq2SeqRNN(nn.Module):
    """
    Modelo completo Seq2Seq con RNN simple (sin atención)
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, config_dict):
        super(Seq2SeqRNN, self).__init__()
        
        embedding_dim = config_dict['embedding_dim']
        hidden_dim = config_dict['hidden_dim']
        num_layers = config_dict['num_layers']
        dropout = config_dict['dropout']
        
        self.encoder = RNNEncoder(
            src_vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout
        )
        
        self.decoder = RNNDecoder(
            tgt_vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout
        )
        
        self.tgt_vocab_size = tgt_vocab_size
    
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
            teacher_forcing_ratio: Probabilidad de usar el ground truth
        
        Returns:
            outputs: (batch_size, tgt_len, vocab_size)
        """
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        
        # Encoder
        _, hidden = self.encoder(src, src_lengths)
        
        # Preparar outputs
        outputs = torch.zeros(
            batch_size, tgt_len, self.tgt_vocab_size
        ).to(tgt.device)
        
        # Primer input del decoder es <bos>
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch, 1)
        
        # Decodificación paso a paso
        for t in range(1, tgt_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t] = output
            
            # Teacher forcing: usar ground truth o predicción
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
    
    def generate(self, src, max_length=config.MAX_LENGTH, device=config.DEVICE):
        """
        Generación greedy (para inferencia)
        
        Args:
            src: (batch_size, src_len)
            max_length: Longitud máxima a generar
        
        Returns:
            outputs: (batch_size, max_length)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Encoder
            _, hidden = self.encoder(src)
            
            # Inicializar con <bos>
            decoder_input = torch.full(
                (batch_size, 1), config.BOS_IDX, dtype=torch.long
            ).to(device)
            
            outputs = [decoder_input]
            
            # Generar token por token
            for _ in range(max_length - 1):
                output, hidden = self.decoder(decoder_input, hidden)
                top1 = output.argmax(1).unsqueeze(1)
                outputs.append(top1)
                
                decoder_input = top1
                
                # Parar si todos generaron <eos>
                if (top1 == config.EOS_IDX).all():
                    break
            
            return torch.cat(outputs, dim=1)

def create_rnn_model(src_vocab_size, tgt_vocab_size):
    """Factory function para crear el modelo RNN"""
    model = Seq2SeqRNN(src_vocab_size, tgt_vocab_size, config.RNN_CONFIG)
    return model

# Test del modelo
if __name__ == "__main__":
    # Parámetros de prueba
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 4
    src_len = 15
    tgt_len = 12
    
    # Crear modelo
    model = create_rnn_model(src_vocab_size, tgt_vocab_size)
    
    # Datos de prueba
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # Forward pass
    outputs = model(src, tgt, teacher_forcing_ratio=0.5)
    
    print("✅ Modelo RNN Simple:")
    print(f"   Input source shape: {src.shape}")
    print(f"   Input target shape: {tgt.shape}")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test de generación
    generated = model.generate(src, max_length=20)
    print(f"   Generated shape: {generated.shape}")
