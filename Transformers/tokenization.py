import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        for token in special_tokens:
            self.word_to_id[token] = self.vocab_size
            self.id_to_word[self.vocab_size] = token
            self.vocab_size += 1

        all_words = []
        for text in texts:
            all_words.extend(text.split())

        for word in sorted(set(all_words)):
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1
            
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """

        ids = []

        ids.append(self.word_to_id[self.bos_token])

        words = text.split()

        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            ids.append(token_id)

        ids.append(self.word_to_id[self.eos_token])

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        
        words = []

        for idx in ids:
            word = self.id_to_word.get(idx, self.unk_token)

            if word in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            
            words.append(word)

        return " ".join(words)
