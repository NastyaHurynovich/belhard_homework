import logging
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import LSTMModel
from sklearn.model_selection import train_test_split

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_training')


with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

input_size = len(X_train[0])
hidden_size = 128
output_size = len(tags)
num_layers = 2
learning_rate = 0.001
num_epochs = 700
print(input_size, output_size)


# Создание датасета
class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]


# Создание DataLoader
train_dataset = ChatDataset(X_train, y_train)
test_dataset = ChatDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

logger.info(f"Критерий: {criterion.__class__.__name__}")
logger.info(f"Оптимизатор: {optimizer.__class__.__name__} с lr={optimizer.param_groups[0]['lr']}")


def evaluate(model, data_loader, criterion, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for (words, labels) in data_loader:
            words = words.to(device)
            labels = labels.to(device)

            # Преобразуем входные данные в torch.float32
            words = words.float()

            # Прямой проход
            outputs = model(words)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)




# Обучение модели с логированием
logger.info(f"Начало обучения на {num_epochs} эпох")
logger.info(f"Размер обучающего набора: {len(train_loader.dataset)} примеров")
logger.info(f"Размер тестового набора: {len(test_loader.dataset)} примеров")
logger.info(f"Параметры обучения: batch_size={train_loader.batch_size}, lr={optimizer.param_groups[0]['lr']}")

# Обучение модели
for epoch in range(num_epochs):
    model.train()
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        words = words.float()

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        train_acc, train_loss = evaluate(model, train_loader, criterion, device)
        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        # Логирование метрик
        logger.info(
            f"Эпоха [{epoch + 1}/{num_epochs}]\n"
            f"  Обучающая выборка: Точность = {train_acc:.4f}, Потеря = {train_loss:.4f}\n"
            f"  Тестовая выборка: Точность = {test_acc:.4f}, Потеря = {test_loss:.4f}"
        )

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

logger.info(f"Обучение завершено. Финальная потеря: {loss.item():.4f}")
print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'file saved to {FILE}')
