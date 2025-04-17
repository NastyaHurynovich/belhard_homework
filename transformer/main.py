import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter, OrderedDict
from tqdm import tqdm  # для прогресс-бара
from torch.cuda.amp import GradScaler, autocast  # для mixed precision


import torch

# Проверяем, доступна ли CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Если CUDA недоступна, выводим предупреждение
if device.type == 'cpu':
    print("⚠️ CUDA not available! Training on CPU (slow).")


class PoetDataset(Dataset):
    def __init__(self, json_path, max_length=64, lowercase=True):  # Уменьшил max_length до 64
        self.max_length = max_length
        self.lowercase = lowercase

        with open(json_path, 'r', encoding='utf-8') as f:
            self.poems = json.load(f)

        self.authors = list(OrderedDict.fromkeys(poem['poet_id'] for poem in self.poems))
        self.author_to_id = {author: i for i, author in enumerate(self.authors)}

        # Добавляем author_id к каждому стихотворению
        for poem in self.poems:
            poem['author_id'] = self.author_to_id[poem['poet_id']]

        text = "\n".join([poem['content'] for poem in self.poems])
        text = self.clean_text(text)

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.encoded = [self.stoi[ch] for ch in text]
        self.encoded = torch.tensor(self.encoded, dtype=torch.long)

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\-\'"\n]', '', text)
        return text.lower() if self.lowercase else text

    def __len__(self):
        return len(self.encoded) - self.max_length

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.max_length + 1]
        return chunk[:-1], chunk[1:]


class PoetTransformer(nn.Module):
    def __init__(self, vocab_size, num_authors, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.author_embed = nn.Embedding(num_authors, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, author_ids=None, mask=None):
        B, T = x.shape  # Batch, Time (sequence length)
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)

        if author_ids is not None:
            author_emb = self.author_embed(author_ids).unsqueeze(1)  # (B, 1, D)
            x = x + author_emb

        if mask is None:
            mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(device)

        x = self.transformer(x, mask)
        return self.fc(x)

    def generate(self, start_text, author_id=None, max_len=100, temperature=1.0, top_k=40):
        self.eval()
        with torch.no_grad():
            input_ids = torch.tensor(
                [self.dataset.stoi[ch] for ch in start_text],
                device=device
            ).unsqueeze(0)

            if author_id is not None:
                author_tensor = torch.tensor([author_id], device=device)
            else:
                author_tensor = None

            for _ in range(max_len):
                seq_len = input_ids.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)

                with autocast():  # Mixed precision для генерации
                    logits = self(input_ids, author_tensor, mask)[0, -1, :] / temperature

                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(0, indices, values)

                probs = torch.softmax(logits, dim=-1)
                next_char = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_char.unsqueeze(0)], dim=1)

            return ''.join([self.dataset.itos[i] for i in input_ids[0].tolist()])


def train_poet_model(json_path):
    dataset = PoetDataset(json_path, max_length=64)  # Уменьшил длину последовательности
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)  # Увеличил batch_size

    model = PoetTransformer(
        vocab_size=dataset.vocab_size,
        num_authors=len(dataset.authors),
        d_model=128,  # Можно уменьшить до 128, если всё ещё медленно
        nhead=4,
        num_layers=4  # Уменьшил число слоёв
    ).to(device)
    model.dataset = dataset

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = GradScaler()  # Для mixed precision
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with autocast():  # Mixed precision
                output = model(x, None)
                loss = criterion(output.view(-1, dataset.vocab_size), y.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"\nEpoch {epoch}, Avg Loss: {total_loss / len(dataloader):.4f}")

        # Генерация только после эпохи (не на каждом батче)
        if epoch % 2 == 0:  # Генерируем раз в 2 эпохи
            print("\nПример генерации:")
            for author in dataset.authors[:2]:
                print(f"Автор {author}:")
                print(model.generate("Луна ", dataset.author_to_id[author], max_len=50))
            print("-" * 50)

    torch.save(model.state_dict(), 'poet_transformer_fast.pth')


if __name__ == "__main__":
    train_poet_model('classic_poems.json')