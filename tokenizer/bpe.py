import regex as re
import unicodedata

class BytePairTokenizer:
    def __init__(self):
         self.cleaner_agent = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d
            | ?\p{L}+
            | ?\p{N}+
            | ?[^\s\p{L}\p{N}]+
            |\s+(?!\S)
            |\s+
            """,
            re.VERBOSE | re.IGNORECASE
        )
         
         #it is unecessary to have complex BPE rules for this example

    def words_to_bytes(self, text):
        # Convert text to bytes
        byte_sequence = text.encode('utf-8')
        # Convert bytes to list of integers
        tokens = list(map(int, byte_sequence))
        return tokens
    
    def text_to_bytes(self, text):
        # Normalize text to NFKC form like é to e +  ́
        text =unicodedata.normalize("NFKC", text)

        # Use regex to find words and punctuation
        words = self.cleaner_agent.findall(text)

        # Convert each word to bytes and flatten the list
        byte_tokens = []
        for word in words:
            byte_tokens.extend(self.words_to_bytes(word))
        return byte_tokens
    
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
    
    def replace_most_common(ids , pairs ,idx_to_token):  
        """Finds the most common byte pair in the list of tokens."""
        new_tokens = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and  ids[i] == pairs[0] and ids[i + 1] == pairs[1]:
                new_tokens.append(idx_to_token)
                i += 2  # Skip the next token as it's merged
            else:
                new_tokens.append(ids[i])
                i += 1
        return new_tokens

    def encode_text(self ,text , merges):
        """Encodes the input text using the BPE merges."""
        byte_tokens = list(map(int, text.encode('utf-8')))
        for pair, idx in merges.items():
            byte_tokens = self.replace_most_common(byte_tokens, pair, idx)
        return byte_tokens
    

    def decode_tokens(tokens, merges):
        """Decodes the list of token ids back to text using the BPE merges."""
        reverse_merges = {idx: pair for pair, idx in merges.items()}

        def expand(token):
            # If token is a merged id, expand recursively
            if token in reverse_merges:
                left, right = reverse_merges[token]
                return expand(left) + expand(right)
            else:
                return [token]

        decoded_tokens = []
        for token in tokens:
            decoded_tokens.extend(expand(token))

        # Now all tokens are <256, safe for bytes()
        byte_sequence = bytes(decoded_tokens)
        return byte_sequence.decode('utf-8', errors='ignore')      
    

