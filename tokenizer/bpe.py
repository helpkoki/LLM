import regex as re

class BytePairTokenizer:
    def __init__(self):
        pass

    def words_to_bytes(self, text):
        # Convert text to bytes
        byte_sequence = text.encode('utf-8')
        # Convert bytes to list of integers
        tokens = list(map(int, byte_sequence))
        return tokens
    
    #Given a list of byte tokens , merge adjacent byte tokens based on BPE rules(common pairs)
    def  merge_tokens(self ,tokens):
        i = 0
        merged_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.bpe_rules:
                merged_token = self.bpe_rules[(tokens[i], tokens[i + 1])]
                merged_tokens.append(merged_token)
                i += 2  # Skip the next token as it's merged
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    #i have learned the i need to make sure that i dont have tokens of dog. and dog! 

    #create a dictionary of byte pairs and their frequencies
    def bytes_pair(self ,tokens):
          pairs = {}
          for i in range(len(tokens) - 1):
              pair = (tokens[i], tokens[i + 1])
              if pair in pairs:
                  pairs[pair] += 1
              else:
                  pairs[pair] = 1
          return pairs         
    

