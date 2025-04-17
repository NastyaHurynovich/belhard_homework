import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, OrderedDict
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import signal
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PoetDataset(Dataset):
    def __init__(self, json_path, max_length=32, lowercase=True):
        self.max_length = max_length
        self.lowercase = lowercase

        with open(json_path, 'r', encoding='utf-8') as f:
            self.poems = json.load(f)

        self.authors = list(OrderedDict.fromkeys(poem['poet_id'] for poem in self.poems))
        self.author_to_id = {author: i for i, author in enumerate(self.authors)}

        for poem in self.poems:
            poem['author_id'] = self.author_to_id[poem['poet_id']]

        text = "\n".join([poem['content'] for poem in self.poems])
        text = self.clean_text(text)
        self.words = self.tokenize(text)
        self.word_counts = Counter(self.words)

        self.vocab = ['<PAD>', '<SOS>', '<EOS>'] + \
                     [word for word, count in self.word_counts.most_common() if count >= 3]

        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        self.encoded = [self.word2idx[word]
                       for word in self.words
                       if word in self.word2idx]
        self.encoded = torch.tensor(self.encoded, dtype=torch.long)

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        tokens = word_tokenize(text, language='russian')
        return tokens

    def clean_text(self, text):
        punctuation = set('.,!?;:"\'()-—')
        cleaned = []
        for char in text:
            if char.isalnum() or char.isspace() or char in punctuation:
                cleaned.append(char)
        return ''.join(cleaned)

    def __len__(self):
        return len(self.encoded) - self.max_length

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.max_length + 1]
        return chunk[:-1], chunk[1:]

class PoetTransformer(nn.Module):
    def __init__(self, vocab_size, num_authors, d_model=256, nhead=8, num_layers=4, max_seq_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.author_embed = nn.Embedding(num_authors, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, author_ids=None, mask=None):
        B, T = x.shape
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)

        if author_ids is not None:
            author_emb = self.author_embed(author_ids).unsqueeze(1)
            x = x + author_emb

        if mask is None:
            mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(device)

        x = self.transformer(x, mask)
        return self.fc(x)

    def generate(self, start_text, author_id=None, max_len=50, temperature=1.0, top_k=40):
        self.eval()
        with torch.no_grad():
            tokens = self.dataset.tokenize(start_text)
            input_ids = torch.tensor(
                [self.dataset.word2idx[token]
                 for token in tokens
                 if token in self.dataset.word2idx],
                device=device
            ).unsqueeze(0)

            if input_ids.size(1) == 0:
                input_ids = torch.tensor([[self.dataset.word2idx['<SOS>']]], device=device)

            if author_id is not None:
                author_tensor = torch.tensor([author_id], device=device)
            else:
                author_tensor = None

            for _ in range(max_len):
                seq_len = input_ids.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)

                logits = self(input_ids, author_tensor, mask)[0, -1, :] / temperature

                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(0, indices, values)

                probs = torch.softmax(logits, dim=-1)
                next_word = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_word.unsqueeze(0)], dim=1)

                # Остановка если сгенерировали <EOS>
                if next_word.item() == self.dataset.word2idx['<EOS>']:
                    break

            words = [self.dataset.idx2word[idx.item()] for idx in input_ids[0]]
            return ' '.join(words)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
        'vocab_size': model.dataset.vocab_size,
        'num_authors': len(model.dataset.authors),
        'word2idx': model.dataset.word2idx,
        'idx2word': model.dataset.idx2word
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, json_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        dataset = PoetDataset(json_path, max_length=32)

        model = PoetTransformer(
            vocab_size=checkpoint['vocab_size'],
            num_authors=checkpoint['num_authors'],
            d_model=256,
            nhead=8,
            num_layers=4
        ).to(device)
        model.dataset = dataset
        model.dataset.word2idx = checkpoint['word2idx']
        model.dataset.idx2word = checkpoint['idx2word']
        model.load_state_dict(checkpoint['model_state'])

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f}")
        return model, optimizer, checkpoint['epoch'], checkpoint['loss']
    return None, None, 0, float('inf')

def handle_interrupt(signal, frame):
    print("\nTraining interrupted! Saving checkpoint...")
    if 'model' in globals() and 'optimizer' in globals() and 'epoch' in globals() and 'current_loss' in globals():
        save_checkpoint(model, optimizer, epoch, current_loss)
    else:
        print("Cannot save checkpoint - training variables not initialized")
    exit(0)

def train_poet_model(json_path, resume_from=None):
    global model, optimizer, epoch, current_loss

    current_loss = float('inf')  # Инициализация переменной

    if resume_from:
        model, optimizer, start_epoch, best_loss = load_checkpoint(resume_from, json_path)
        if model is None:
            print("No valid checkpoint found. Starting from scratch.")
            return
    else:
        dataset = PoetDataset(json_path, max_length=32)
        model = PoetTransformer(
            vocab_size=dataset.vocab_size,
            num_authors=len(dataset.authors),
            d_model=256,
            nhead=8,
            num_layers=4
        ).to(device)
        model.dataset = dataset
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        start_epoch = 0
        best_loss = float('inf')

    signal.signal(signal.SIGINT, handle_interrupt)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    dataloader = DataLoader(model.dataset, batch_size=128, shuffle=True)

    try:
        for epoch in range(start_epoch, 5):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

            for batch_idx, (x, y) in enumerate(progress_bar):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                with autocast():
                    output = model(x)
                    loss = criterion(output.view(-1, model.dataset.vocab_size), y.view(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                current_loss = loss.item()
                total_loss += current_loss
                progress_bar.set_postfix(loss=current_loss)

            avg_loss = total_loss / len(dataloader)
            current_loss = avg_loss  # Обновляем current_loss для использования в except
            print(f"\nEpoch {epoch}, Avg Loss: {avg_loss:.4f}")

            save_checkpoint(model, optimizer, epoch, avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'best_model_1.pth')

            if epoch % 2 == 0:
                print("\nПример генерации:")
                for author in model.dataset.authors[:2]:
                    print(f"Автор {author}:")
                    print(model.generate("люблю ", model.dataset.author_to_id[author], max_len=20))
                print("-" * 50)

    except Exception as e:
        print(f"\nError during training: {e}")
        save_checkpoint(model, optimizer, epoch, current_loss)
        raise

if __name__ == "__main__":
    checkpoint_dir = "checkpoints"
    last_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')])
        if checkpoints:
            last_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

    train_poet_model('classic_poems.json', resume_from=last_checkpoint)