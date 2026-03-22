import torch
import torch.nn as nn
from .config import KalanidhiConfig

class KalanidhiModel(nn.Module):
    def __init__(self, config: KalanidhiConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Time embedding: expects [batch, 1] input
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

        # Output head with weight tying
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

def forward(self, input_ids, t, attention_mask=None):
        # t: [batch] or [batch, 1] -> normalize to [batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Time embedding: [batch, 1, hidden]
        t_emb = self.time_mlp(t).unsqueeze(1)

        # Token + position embeddings
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.embed_dropout(x)

        # Convert HuggingFace mask (1=keep, 0=ignore) to PyTorch format (True=ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        # Add time context and pass through transformer
        x = self.transformer(x + t_emb, src_key_padding_mask=src_key_padding_mask)

        return self.lm_head(x)