import random
import numpy as np

class DataLoader:
      def __init__(self, dataset, batch_size, shuffle=True):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0

        def reset(self):
            self.current_idx = 0
            if self.shuffle:
                random.shuffle(self.indices)
        
        def __iter__(self):
            self.reset()
            return self
        
        def __next__(self):
            if self.current_idx >= len(self.indices):
                raise StopIteration

            batch_indices = self.indices[
                self.current_idx : self.current_idx + self.batch_size
            ]
            self.current_idx += self.batch_size

            x_batch = []
            y_batch = []

            for idx in batch_indices:
                x, y = self.dataset[idx]
                x_batch.append(x)
                y_batch.append(y)

            return np.array(x_batch), np.array(y_batch)