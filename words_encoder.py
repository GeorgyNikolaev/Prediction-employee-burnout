import torch.nn as nn
import torch


class WordEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads=4):
        super(WordEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.keys = nn.Parameter(torch.Tensor(2 * embedding_dim, embedding_dim))
        self.keys.data.uniform_(-1.0, 1.0)
        self.queries = nn.Parameter(torch.Tensor(2 * embedding_dim, embedding_dim))
        self.queries.data.uniform_(-1.0, 1.0)
        self.values = nn.Parameter(torch.Tensor(2 * embedding_dim, embedding_dim))
        self.values.data.uniform_(-1.0, 1.0)
        self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        embed = self.embedding(x)
        pos_embed = self.positional_embedding(x)
        x = embed + pos_embed
        key, query, value = self.keys(x), self.queries(x), self.values(x)
        x = self.attention_layer(key, query, value)
        x = self.feed_forward(x)
        return x
