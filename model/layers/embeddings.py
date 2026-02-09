import numpy as np

class SimpleEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim

        # Initialize embedding matrix with random values
        self.embeddings = np.random.rand(vocab_size, embedding_dim)

    def forward(self, input_indices):
        return self.embeddings[input_indices]
    
  

class SimplePositionalEncoding:
    def __init__(self, max_len, embed_dim):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.positional_encoding = self._create_positional_encoding()

    def _create_positional_encoding(self):
        positional_encoding = np.zeros((self.max_len, self.embed_dim))
        for pos in range(self.max_len):
            for i in range(0, self.embed_dim, 2):
                positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.embed_dim)))
                if i + 1 < self.embed_dim:
                    positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / self.embed_dim)))
        return positional_encoding

    def forward(self, input_indices):
        batch_size, seq_len = input_indices.shape
        return self.positional_encoding[:seq_len]