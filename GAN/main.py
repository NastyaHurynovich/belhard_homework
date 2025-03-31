import os
import torch
from torchvision import transforms
from PIL import Image
from models import GeneratorUNet


def process_sketch(input_path, output_path, model_path="D:/GAN/saved_models/facades/generator_190.pth"):
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Загрузка модели
        generator = GeneratorUNet().to(device)
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()

        # Преобразования
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = Image.open(input_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = generator(image_tensor)

        # Сохранение
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu() * 0.5 + 0.5)
        output_image.save(output_path)
        print(f"Success! Result saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input sketch")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--model", type=str, default="D:/GAN/saved_models/facades/generator_190.pth",
                        help="Path to trained model")
    args = parser.parse_args()

    process_sketch(args.input, args.output, args.model)