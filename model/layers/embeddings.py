import numpy as np

class SimpleEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim

        # Initialize embedding matrix with random values
        self.embeddings = np.random.rand(vocab_size, embedding_dim)

    def forward(self, input_indices):
        return self.embeddings[input_indices]