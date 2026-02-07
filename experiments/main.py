from LLM.tokenizer import bpe
from LLM.data import dataset

tokenizer = bpe.BPETokenizer()
#C:\Users\koket\Desktop\project\CS50\AI\LLM\LLM\data\raw\taylorswift.txt
# text = tokenizer.document_to_string('LLM/data/raw/taylorswift.txt')
# ids , merges = tokenizer.train(text, 1000)

store_merges = 'LLM/data/cleaned/taylorswiftmerges.json'
# tokenizer.store_merges(store_merges)
# tokenizer.create_file_with_merges_and_tokens(merges, ids, 'LLM/data/cleaned/taylorswift_tokens_and_merges.txt')
# tokenizer.load_merges(store_merges)
# ids_from_word = tokenizer.encode("Taylor Swift is a singer-songwriter. 你好，我的名字是")
# print(ids_from_word)
# decoded_text = tokenizer.decode(ids_from_word)  
# print(decoded_text)

tokens = [1, 2, 3, 4, 5, 6]
datasetx = dataset.Dataset(tokens, 3)

for i in range(len(datasetx)):
    x, y = datasetx[i]
    print(x, "->", y)
