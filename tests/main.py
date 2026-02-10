from LLM.model.layers import attention
import numpy as np


attention_layer = attention.SimpleSelfAttention(embed_dim=4)
input_data = np.random.rand(2, 3, 4)  # batch_size=2, seq_len=3, embed_dim=4
output = attention_layer.forward(input_data)
print("Input shape:", input_data.shape)
print("Output shape:", output.shape)


# from LLM.model.layers import embeddings
# embedding_layer = embeddings.SimpleEmbedding(vocab_size=10, embedding_dim=4)
#Sanity Test
from LLM.model.layers import embeddings
vocab_size = 100
embed_dim = 16

embedding = embeddings.SimpleEmbedding(vocab_size, embed_dim)

tokens = np.array([
    [1, 5, 9],
    [4, 2, 7]
])

out = embedding.forward(tokens)

print(out.shape)
