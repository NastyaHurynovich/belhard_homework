import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasets import ImageDataset
from models import GeneratorUNet, Discriminator, weights_init_normal


torch.backends.cudnn.benchmark = True


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--dataset_name", type=str, default="facades")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--img_height", type=int, default=128)
    parser.add_argument("--img_width", type=int, default=128)
    parser.add_argument("--channels", type=int, default=3)
    opt = parser.parse_args()


    patch = (1, 15, 15)
    lambda_pixel = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(f"images/{opt.dataset_name}", exist_ok=True)
    os.makedirs(f"saved_models/{opt.dataset_name}", exist_ok=True)

    print(
        f"Using device: {device}, GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2:.0f}MB")

    # Модель
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # Веса
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Функция потерь
    criterion_GAN = nn.MSELoss().to(device)
    criterion_pixelwise = nn.L1Loss().to(device)

    # Оптимизаторы
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Трансформы и DataLoader
    transforms_ = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataloader = DataLoader(
        ImageDataset(root="D:\\GAN\\data\\facades", transforms_=transforms_, mode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Обучение
    with torch.no_grad():
        test_input = torch.randn(1, 3, opt.img_height, opt.img_width).to(device)
        test_output = discriminator(test_input, test_input)
        patch = test_output.shape[-2:]  # Автоматическое определение размера патча
        print(f"Размер выхода дискриминатора: {patch}")

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch["B"].to(device)
            real_B = batch["A"].to(device)

            valid = torch.ones((real_A.size(0), 1, *patch), device=device)
            fake = torch.zeros((real_A.size(0), 1, *patch), device=device)

            # Обучение генератора
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # Обучение дискриминатора
            optimizer_D.zero_grad()
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()



            batches_done = epoch * len(dataloader) + i
            if batches_done % 100 == 0:
                print(
                    f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]"
                )

            # Сохранение примеров
            if batches_done % 500 == 0:
                with torch.no_grad():
                    fake_B = generator(real_A[:5])  # Берем первые 5 изображений
                    img_sample = torch.cat((real_A[:5].data, fake_B.data, real_B[:5].data), -2)
                    save_image(img_sample, f"images/{opt.dataset_name}/{batches_done}.png", nrow=5, normalize=True)

        # Сохранение моделей
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"saved_models/{opt.dataset_name}/generator_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"saved_models/{opt.dataset_name}/discriminator_{epoch}.pth")


if __name__ == "__main__":
    try:
        import numpy as np

        if np.__version__.startswith('2'):
            print("Error: NumPy 2.x detected, please downgrade to 1.x")
            print("Run: pip install 'numpy<2'")
            exit()
    except ImportError:
        pass

    main()