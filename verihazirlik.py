from pathlib import Path
import torch


class Verihazirlik:
    def __init__(self):
        self.input_tensor = []

        # özel tokenlar
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        self.word2id = {token: idx for idx, token in enumerate(special_tokens)}
        self.id2word = {idx: token for idx, token in enumerate(special_tokens)}

        self.current_id = len(special_tokens)
        self.pad_id = self.word2id["<PAD>"]

        # 🔥 max uzunluk (OOM çözümü)
        self.MAX_LEN = 64

    # 📂 dosya okuma
    def dosyaokuma(self):
        path = Path("datalar/aiveri.txt")
        text = path.read_text(encoding="utf-8")

        # bloklara ayır
        blocks = [b.strip() for b in text.split("[p]") if b.strip()]

        tum_cumleler = []

        for block in blocks:
            block = block.replace("[/p]", "").strip()

            # cümlelere böl
            cumleler = [c.strip() for c in block.split(".") if c.strip()]

            tum_cumleler.append(cumleler)

        print(tum_cumleler[0])
        return tum_cumleler

    # 🧠 tokenize + pad
    def tokenize_and_pad(self, cumleler):
        self.input_tensor = []

        # 1️⃣ vocab oluştur
        for cumle in cumleler:
            for sentence in cumle:
                for kelime in sentence.split():
                    if kelime not in self.word2id:
                        self.word2id[kelime] = self.current_id
                        self.id2word[self.current_id] = kelime
                        self.current_id += 1

        print("Vocab size:", len(self.word2id))

        # 2️⃣ tokenize + pad
        for cumle in cumleler:
            for sentence in cumle:

                tokens = sentence.split()

                ids = [self.word2id.get(k, self.word2id["<UNK>"]) for k in tokens]

                # BOS + EOS ekle
                ids = [self.word2id["<BOS>"]] + ids + [self.word2id["<EOS>"]]

                # 🔥 kes (OOM engelle)
                ids = ids[:self.MAX_LEN]

                # 🔥 pad
                ids = ids + [self.pad_id] * (self.MAX_LEN - len(ids))

                self.input_tensor.append(ids)

        return torch.tensor(self.input_tensor, dtype=torch.long), self.word2id, self.id2word

    # 🎯 x-y oluştur (next token prediction)
    def makex_y(self, input_tensor):
        x = input_tensor[:, :-1]
        y = input_tensor[:, 1:]
        #print(x[0])
        return x, y

    # 🔢 vocab size
    def getvocabsize(self):
        return len(self.word2id)