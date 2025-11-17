"""
MODELO 2: LSTM con Atención Bahdanau
Arquitectura encoder-decoder con LSTM bidireccional y mecanismo de atención
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class BahdanauAttention(nn.Module):
    """
    Mecanismo de atención Bahdanau (additive attention)
    
    score(h_t, h_s) = v^T * tanh(W_1 * h_t + W_2 * h_s)
    """
    
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Transformaciones lineales
        self.W_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Vector de score
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, hidden_dim) - Estado actual del decoder
            encoder_outputs: (batch, src_len, hidden_dim) - Salidas del encoder
            mask: (batch, src_len) - Máscara de padding (True donde hay padding)
        
        Returns:
            context: (batch, hidden_dim) - Vector de contexto ponderado
            attention_weights: (batch, src_len) - Pesos de atención
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Expandir decoder_hidden para broadcast
        # (batch, hidden) -> (batch, 1, hidden) -> (batch, src_len, hidden)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calcular scores de atención
        # energy: (batch, src_len, hidden)
        energy = torch.tanh(
            self.W_decoder(decoder_hidden) + self.W_encoder(encoder_outputs)
        )
        
        # attention_scores: (batch, src_len)
        attention_scores = self.v(energy).squeeze(2)
        
        # Aplicar máscara (poner -inf donde hay padding)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e10)
        
        # Normalizar con softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, src_len)
        
        # Calcular contexto como suma ponderada
        # context: (batch, hidden_dim)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs  # (batch, src_len, hidden)
        ).squeeze(1)  # (batch, hidden)
        
        return context, attention_weights

class LSTMEncoder(nn.Module):
    """
    Encoder con LSTM bidireccional
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        
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
        
        # LSTM
        self.lstm = nn.LSTM(
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
            self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src, src_lengths=None):
        """
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
        
        Returns:
            outputs: (batch, src_len, hidden_dim) - si bidireccional ya proyectado
            hidden: (num_layers, batch, hidden_dim)
            cell: (num_layers, batch, hidden_dim)
        """
        # Embedding
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb_dim)
        
        # LSTM
        # outputs: (batch, src_len, hidden*num_directions)
        # hidden: (num_layers*num_directions, batch, hidden)
        # cell: (num_layers*num_directions, batch, hidden)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Si es bidireccional, combinar direcciones
        if self.bidirectional:
            # outputs ya está concatenado (forward + backward)
            # Proyectar a hidden_dim
            outputs = self.fc_hidden(outputs)  # (batch, src_len, hidden_dim)
            
            # Para hidden y cell: combinar capas forward y backward
            # hidden: (num_layers*2, batch, hidden) -> (num_layers, batch, hidden*2)
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # (num_layers, batch, hidden*2)
            hidden = self.fc_hidden(hidden)  # (num_layers, batch, hidden_dim)
            
            cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
            cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
            cell = self.fc_cell(cell)
        
        return outputs, hidden, cell

class LSTMAttentionDecoder(nn.Module):
    """
    Decoder con LSTM y mecanismo de atención Bahdanau
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(LSTMAttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=config.PAD_IDX
        )
        
        # Atención
        self.attention = BahdanauAttention(hidden_dim)
        
        # LSTM (input = embedding + context)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,  # embedding + context
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Capa de salida (context + hidden + embedding)
        self.fc_out = nn.Linear(
            hidden_dim + hidden_dim + embedding_dim,
            vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, hidden, cell, encoder_outputs, src_mask=None):
        """
        Args:
            tgt: (batch, 1) - Token actual
            hidden: (num_layers, batch, hidden_dim)
            cell: (num_layers, batch, hidden_dim)
            encoder_outputs: (batch, src_len, hidden_dim)
            src_mask: (batch, src_len)
        
        Returns:
            output: (batch, vocab_size)
            hidden: (num_layers, batch, hidden_dim)
            cell: (num_layers, batch, hidden_dim)
            attention_weights: (batch, src_len)
        """
        # Embedding
        embedded = self.dropout(self.embedding(tgt))  # (batch, 1, emb_dim)
        
        # Calcular atención usando el último hidden state
        # hidden[-1]: (batch, hidden_dim)
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs, src_mask
        )
        
        # Concatenar embedding con context
        # context: (batch, hidden_dim) -> (batch, 1, hidden_dim)
        context = context.unsqueeze(1)
        lstm_input = torch.cat([embedded, context], dim=2)  # (batch, 1, emb+hidden)
        
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch, 1, hidden_dim)
        
        # Preparar output final: concatenar embedding, context, hidden
        output = output.squeeze(1)  # (batch, hidden_dim)
        embedded = embedded.squeeze(1)  # (batch, emb_dim)
        context = context.squeeze(1)  # (batch, hidden_dim)
        
        output = self.fc_out(
            torch.cat([output, context, embedded], dim=1)
        )  # (batch, vocab_size)
        
        return output, hidden, cell, attention_weights

class Seq2SeqLSTMAttention(nn.Module):
    """
    Modelo completo Seq2Seq con LSTM y Atención Bahdanau
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, config_dict):
        super(Seq2SeqLSTMAttention, self).__init__()
        
        embedding_dim = config_dict['embedding_dim']
        hidden_dim = config_dict['hidden_dim']
        num_layers = config_dict['num_layers']
        dropout = config_dict['dropout']
        bidirectional = config_dict.get('bidirectional', True)
        
        self.encoder = LSTMEncoder(
            src_vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout,
            bidirectional
        )
        
        self.decoder = LSTMAttentionDecoder(
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
        
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Preparar outputs
        outputs = torch.zeros(
            batch_size, tgt_len, self.tgt_vocab_size
        ).to(tgt.device)
        
        # Almacenar pesos de atención
        attention_weights_all = []
        
        # Primer input del decoder es <bos>
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch, 1)
        
        # Decodificación paso a paso
        for t in range(1, tgt_len):
            output, hidden, cell, attn_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, src_mask
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
            encoder_outputs, hidden, cell = self.encoder(src)
            
            # Inicializar con <bos>
            decoder_input = torch.full(
                (batch_size, 1), config.BOS_IDX, dtype=torch.long
            ).to(device)
            
            outputs = [decoder_input]
            attention_weights_all = []
            
            # Generar token por token
            for _ in range(max_length - 1):
                output, hidden, cell, attn_weights = self.decoder(
                    decoder_input, hidden, cell, encoder_outputs, src_mask
                )
                
                top1 = output.argmax(1).unsqueeze(1)
                outputs.append(top1)
                attention_weights_all.append(attn_weights)
                
                decoder_input = top1
                
                # Parar si todos generaron <eos>
                if (top1 == config.EOS_IDX).all():
                    break
            
            return torch.cat(outputs, dim=1), attention_weights_all

def create_lstm_model(src_vocab_size, tgt_vocab_size):
    """Factory function para crear el modelo LSTM con atención"""
    model = Seq2SeqLSTMAttention(src_vocab_size, tgt_vocab_size, config.LSTM_CONFIG)
    return model

# Test del modelo
if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 4
    src_len = 15
    tgt_len = 12
    
    model = create_lstm_model(src_vocab_size, tgt_vocab_size)
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    src_mask = (src == config.PAD_IDX)
    
    outputs, attn_weights = model(src, tgt, src_mask, teacher_forcing_ratio=0.5)
    
    print("✅ Modelo LSTM con Atención Bahdanau:")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Attention weights: {len(attn_weights)} pasos")
    print(f"   Atención shape por paso: {attn_weights[0].shape}")
    print(f"   Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
