import numpy as np

class SimpleSelfAttention:
     def __init__(self , embed_dim):
         self.embed_dim = embed_dim

         self.W_q = np.random.rand(embed_dim, embed_dim) *0.02
         self.W_k = np.random.rand(embed_dim, embed_dim) *0.02
         self.W_v = np.random.rand(embed_dim, embed_dim) *0.02

     def forward(self, x):
          
          batch_size, seq_len, embed_dim = x.shape

          Q = np.dot(x, self.W_q)
          K = np.dot(x, self.W_k)
          V = np.dot(x, self.W_v)

          scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.embed_dim)
          mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
          scores[:, mask] = -1e9
          attention_weights = self._softmax(scores)
          output = np.matmul(attention_weights, V)

          return output
     
     def _softmax(self, x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)