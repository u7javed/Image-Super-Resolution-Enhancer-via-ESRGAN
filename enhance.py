
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

from models import Generator

def main():
    #take in arguments
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')

    # parameters needed to enhance image
    parser.add_argument('--image_file', type=str, default='test_image.png', help='location of image file to be enhanced')
    parser.add_argument('--file_length', type=int, default=128, help='sidelength of of transformation (ideal square = 128)')
    parser.add_argument('--dir_to_generator', type=st, default='best_generator.pt', help='directory to generator to enhance images')
    parser.add_argument('--save_directory', type=str, default='', help='directory where enhanced image will be saved')

    args = parser.parse_args()

    image_file = args.image_file
    save_directory = args.save_directory
    file_length = args.file_length

    #transformation applied to image
    transformation = transforms.Compose([
        transforms.Resize((file_length, file_length), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])


    pil_image = Image.open(image_file)

    #conver to tensor and conver to batch of size 1
    image_tensor = transformation(pil_image)
    image_tensor = image_tensor.view(1, *(image_tensor.size()))

    #load enhancer (generator)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enhancer = Generator(channels=3).to(device)
    enhancer.load_state_dict(torch.load('best_generator_model.pt'))

    enhanced_image = enhancer(image_tensor.to(device))
    image_grid = torchvision.utils.make_grid(enhanced_image.cpu().detach(), nrow=1, normalize=True)
    _, plot = plt.subplots(figsize=(8, 8))
    plt.axis('off')
    plot.imshow(image_grid.permute(1, 2, 0))
    plt.savefig(save_directory + '/enhanced_image.png', bbox_inches='tight')

if __name__ == "__main__":
    main()