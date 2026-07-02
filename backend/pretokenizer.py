from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import pickle
import regex as re

from .tokenizer import BPETokenizer


def _count_pairs_batch(word_batch):
    pairs = Counter()
    for wid in word_batch:
        pairs.update(zip(wid, wid[1:]))
    return pairs


def _apply_merge_batch(word_batch, pair, idx):
    result = []
    for wid in word_batch:
        new_ids = []
        i = 0
        while i < len(wid):
            if i < len(wid) - 1 and pair[0] == wid[i] and pair[1] == wid[i + 1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(wid[i])
                i += 1
        result.append(new_ids)
    return result


class PreBPETokenizer(BPETokenizer):
    def __init__(self, num_workers=None):
        super().__init__()
        self.num_workers = num_workers or (__import__("os").cpu_count() or 4)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _pre_tokenize(self, text: str) -> list[str]:
        return self.pat.findall(text)

    def _batches(self, items, num_chunks):
        chunk_size = max(1, len(items) // max(1, num_chunks))
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    def train(self, corpus, vocab_size, progress_callback=None):
        assert vocab_size > 256, "Vocab Size must be atleast 256"
        self.num_merges = vocab_size - 256

        if self.trained:
            self._clear_state()

        words = self._pre_tokenize(corpus)
        word_ids = [list(w.encode("utf-8")) for w in words if w]

        total = self.num_merges
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            for i in tqdm(range(total), leave=False,
                          desc="Training Tokenizer"):
                if progress_callback:
                    progress_callback(i + 1, total)
                batches = self._batches(word_ids, self.num_workers)

                pairs = Counter()
                for pc in pool.map(_count_pairs_batch, batches):
                    pairs.update(pc)

                if not pairs:
                    break

                most_common_pair = max(pairs, key=pairs.get)
                if pairs.get(most_common_pair) == 1:
                    break

                idx = 256 + i
                word_ids = []
                for batch_result in pool.map(
                    lambda args: _apply_merge_batch(*args),
                    [(b, most_common_pair, idx) for b in batches]
                ):
                    word_ids.extend(batch_result)
                self.merge_rules[most_common_pair] = idx

        self._build_vocab()
        self.trained = True

    def encode(self, text):
        assert self.trained, "Tokenizer has not been trained"
        words = self._pre_tokenize(text)

        def encode_batch(word_batch):
            result = []
            for word in word_batch:
                word_tokens = list(word.encode("utf-8"))
                while len(word_tokens) >= 2:
                    best_pair = None
                    best_rank = float("inf")
                    for pair in zip(word_tokens, word_tokens[1:]):
                        rank = self.merge_rules.get(pair)
                        if rank is not None and rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                    if best_pair is None:
                        break
                    new_ids = []
                    j = 0
                    while j < len(word_tokens):
                        if j < len(word_tokens) - 1 and best_pair[0] == word_tokens[j] and best_pair[1] == word_tokens[j + 1]:
                            new_ids.append(self.merge_rules[best_pair])
                            j += 2
                        else:
                            new_ids.append(word_tokens[j])
                            j += 1
                    word_tokens = new_ids
                result.extend(word_tokens)
            return result

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            batches = self._batches([w for w in words if w], self.num_workers)
            tokens = []
            for batch_result in pool.map(encode_batch, batches):
                tokens.extend(batch_result)
        return tokens
