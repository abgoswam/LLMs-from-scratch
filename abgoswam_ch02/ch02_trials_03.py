
import os
import re
import urllib.request

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
if __name__ == "__main__":

    if not os.path.exists("the-verdict.txt"):
        url = ("https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt")
        file_path = "the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

    all_words = sorted(set(preprocessed))

    vocab = {token:integer for integer,token in enumerate(all_words)}

    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," 
            Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

    text = "Hello, do you like tea?"
    ids = tokenizer.encode(text)
    print(ids)

    print("done")
