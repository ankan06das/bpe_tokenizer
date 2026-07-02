from __future__ import annotations
from collections import Counter
from tqdm.auto import tqdm
import pickle

class BPETokenizer:
    def __init__(self):

        self.merge_rules = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.trained = False
        
    def _clear_state(self):
        self.merge_rules.clear()
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def _get_pair_count(self,ids):
        return Counter(zip(ids, ids[1:]))

    def _apply_merge(self,ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and pair[0] == ids[i] and pair[1] == ids[i+1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _build_vocab(self):
        for (pair_0, pair_1), idx in self.merge_rules.items():
            self.vocab[idx] = self.vocab[pair_0] + self.vocab[pair_1]
        
    def train(self, corpus, vocab_size):
        assert vocab_size > 256, "Vocab Size must be atleast 256"
        self.num_merges = vocab_size - 256
        
        if self.trained:
            self._clear_state()
        ids = list(corpus.encode("utf-8"))
        for i in tqdm(range(self.num_merges),leave=False,
                        desc="Training Tokenizer"):
            pairs = self._get_pair_count(ids)

            if not pairs:
                break

            most_common_pair =  max(pairs, key = pairs.get)
            if pairs.get(most_common_pair) == 1:
                break
            
            idx = 256 + i
            ids = self._apply_merge(ids, most_common_pair, idx)
            self.merge_rules[most_common_pair] = idx

        self._build_vocab()
        
        self.trained = True

    def encode(self, text):
        assert self.trained, "Tokenizer has not been trained"
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            best_pair = None
            best_rank = float("inf")

            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_rules.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            tokens = self._apply_merge(tokens, best_pair, self.merge_rules[best_pair])
        return tokens
    
    def decode(self, ids):
        assert self.trained, "Tokenizer has not been trained"
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors='replace')
        return text

    def tokenize(self, text):
        ids = self.encode(text)
        return [
            {
                "id": idx,
                "token": self.decode([idx])
            }
            for idx in ids
        ]
        
    def save(self, path:str):
        assert path.endswith(".pkl"), "File need to be pickle (.pkl) file"
        with open(path, 'wb') as f:
            pickle.dump(self.merge_rules, f)
    
    def load(self, path:str):
        assert path.endswith(".pkl"), "File need to be pickle (.pkl) file"
        with open(path, "rb") as f:
            self.merge_rules = pickle.load(f)
        self._build_vocab()
        self.trained = True