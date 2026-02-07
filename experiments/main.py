from LLM.tokenizer import bpe
from LLM.data import dataset

tokenizer = bpe.BPETokenizer()
#C:\Users\koket\Desktop\project\CS50\AI\LLM\LLM\data\raw\taylorswift.txt
text = tokenizer.document_to_string('LLM/data/raw/taylorswift.txt')
ids , merges = tokenizer.train(text, 1000)

store_merges = 'LLM/data/cleaned/taylorswiftmerges.json'
tokenizer.store_merges(store_merges)
tokenizer.create_file_with_merges_and_tokens(merges, ids, 'LLM/data/cleaned/taylorswift_tokens_and_merges.txt')
print(text[:100])
