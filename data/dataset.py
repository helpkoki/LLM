

class Dataset:
    def __init__(self, tokens , seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
       

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len+1]
        return x, y
    
    #New helper method
    def preview(self):
        print(f"Dataset length: {len(self)}")
        for i in range(len(self)):
            x, y = self[i]
            print(f"x={x}, y={y}")