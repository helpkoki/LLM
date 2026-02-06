import regex as re
import unicodedata

import json


class BPETokenizer:
    def __init__(self):
        self.merges = {}
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

    # ---------------- TRAIN ----------------
    def train(self, text, vocab_size, verbose=False):
        pre_bytes = text.encode("utf-8")
        byte_tokens = list(map(int, pre_bytes))

        num_merges = vocab_size - 256
        ids = list(byte_tokens)
        self.merges = {}

        for i in range(num_merges):
            pairs = self.get_pairs(ids)
            if not pairs:
                break

            pair = max(pairs, key=pairs.get)
            idx = 256 + i
            ids = self.replace_most_common(ids, pair, idx)

            if verbose:
                print(f"Merging {pair} -> {idx}")

            self.merges[pair] = idx

        return ids, self.merges

    # ---------------- FILE HELPERS ----------------
    def create_file(self, text, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for line in text.splitlines():
                f.write(line + "\n")

    def create_file_with_merges(self, merges, ids, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for pair, idx in merges.items():
                f.write(f"{pair} -> {idx}\n")

            f.write("\nFinal token ids:\n")
            f.write(str(ids))

    def create_file_with_merges_and_tokens(self, merges, ids, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== Token IDs ===\n")
            seen_tokens = set()

            for token in ids:
                if token not in seen_tokens:
                    seen_tokens.add(token)
                    decoded = chr(token) if isinstance(token, int) else str(token)
                    f.write(f"{token} -> {decoded}\n")

            f.write("\n=== BPE Merges ===\n")

            seen_pairs = set()
            for pair, new_token_id in merges.items():
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    pair_str = " ".join(
                        chr(p) if isinstance(p, int) else str(p) for p in pair
                    )
                    f.write(f"{pair_str} -> {new_token_id}\n")

    def pre_tokenize(self, text, return_chunks=False):
        chunks = self.cleaner_agent.findall(text)

        if return_chunks:
            return chunks

        return "".join(chunks)


    # ---------------- CORE BPE LOGIC ----------------
    def replace_most_common(self, ids, pair, new_token):
        new_tokens = []
        i = 0

        while i < len(ids):
            if (
                i < len(ids) - 1
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
            ):
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(ids[i])
                i += 1

        return new_tokens

    def get_pairs(self, tokens):
        pairs = {}

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1

        return pairs

    # ---------------- STORE / LOAD MERGES ----------------
    def store_merges(self, filename):
        serialized = {
            f"{pair[0]} {pair[1]}": idx for pair, idx in self.merges.items()
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=4)

    def load_merges(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = {
            tuple(map(int, pair.split())): idx for pair, idx in data.items()
        }

        return self.merges

    # ---------------- ENCODE ----------------
    def encode(self, text):
        byte_tokens = list(map(int, text.encode("utf-8")))

        for pair, idx in self.merges.items():
            byte_tokens = self.replace_most_common(byte_tokens, pair, idx)

        return byte_tokens

    # ---------------- DECODE ----------------
    def decode(self, tokens):
        reverse_merges = {idx: pair for pair, idx in self.merges.items()}

        def expand(token):
            if token in reverse_merges:
                left, right = reverse_merges[token]
                return expand(left) + expand(right)
            return [token]

        decoded_tokens = []
        for token in tokens:
            decoded_tokens.extend(expand(token))

        byte_sequence = bytes(decoded_tokens)
        return byte_sequence.decode("utf-8", errors="ignore")

    # ---------------- DOCUMENT READER ----------------
    def document_to_string(self, document):
        with open(document, "r", encoding="utf-8") as file:
            return file.read()
