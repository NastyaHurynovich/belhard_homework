import torch
from torch.utils.data import dataset

from main import PoetTransformer

# Загрузка модели
model = PoetTransformer(vocab_size, num_authors).to(device)
model.load_state_dict(torch.load('poet_transformer.pth'))
model.dataset = dataset

# Генерация в стиле Пушкина
print(model.generate("Зима ", author_to_id['pushkin'], max_len=200))

# Свободная генерация
print(model.generate("Люблю ", max_len=200))
