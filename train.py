import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os
from glob import glob
from PIL import Image

from models import *


class ResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, low_res_dir, high_res_dir, low_res_length, high_res_length):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir

        # low resolution transformation
        self.low_res_transform = transforms.Compose([
            transforms.Resize((low_res_length, low_res_length),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        # high resolution transformation
        self.high_res_transform = transforms.Compose([
            transforms.Resize((high_res_length, high_res_length),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        # search for png files in respective directories
        self.low_res_list = list(
            map(Image.open, glob(self.low_res_dir + '*.png')))
        self.high_res_list = list(
            map(Image.open, glob(self.high_res_dir + '*.png')))

    def __len__(self):
        return len(self.low_res_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        high_res_image = self.high_res_list[idx]
        low_res_image = self.low_res_list[idx]

        high_res_image = self.high_res_transform(high_res_image)
        low_res_image = self.low_res_transform(low_res_image)

        return low_res_image, high_res_image


class Trainer():

    def __init__(self, lr_dir_train, hr_dir_train, lr_dir_test, hr_dir_test, lr_length, hr_length, batch_size, lambda_adv=5e-3, lambda_pixel=1e-2, b1=0.9, b2=0.999, channels=3, feature_space=64, device='cpu', lr=0.0002, num_workers=1):

        self.train_dataset = ResolutionDataset(lr_dir_train, hr_dir_train, lr_length, hr_length)
        self.test_dataset = ResolutionDataset(lr_dir_test, hr_dir_test, lr_length, hr_length)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.batch_size = batch_size
        self.device = device
        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel

        self.gen = Generator(channels, feature_space).to(self.device)
        self.dis = Discriminator(hr_length, channels, feature_space).to(self.device)
        self.ext = FeatureExtractor().to(self.device)

        self.loss_func_logits = nn.BCEWithLogitsLoss().to(self.device)
        self.content_loss_func = nn.L1Loss().to(self.device)
        self.pixel_loss_func = nn.L1Loss().to(self.device)

        self.optimizer_g = optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_d = optim.Adam(self.dis.parameters(), lr=lr, betas=(b1, b2))

    def train(self, epochs, saved_image_directory, saved_model_directory):
        start_time = time.time()

        gen_loss_list = []
        dis_loss_list = []

        for epoch in range(epochs):
            cur_time = time.time()
            for i, (low_res_images, high_res_images) in enumerate(self.train_loader):
                iterations_completed = epoch*len(self.train_loader) + i

                b_size = len(low_res_images)

                low_res_images = low_res_images.to(self.device)
                high_res_images = high_res_images.to(self.device)

                real = torch.ones((b_size, *self.dis.output_shape)).to(self.device)
                fake = torch.zeros((b_size, *self.dis.output_shape)).to(self.device)

                # train generator
                self.optimizer_g.zero_grad()

                fake_images = self.gen(low_res_images)

                #pixel loss
                p_loss = self.pixel_loss_func(fake_images, high_res_images)

                # train generator over pixel loss for a certain number of iterations before introducing discriminator
                if iterations_completed < 500:
                    p_loss.backward()
                    self.optimizer_g.step()
                    print('     [{}/{}][{}/{}],     Pixel loss: {:.4f}\n'.format(epoch, epochs, i, len(self.train_loader), p_loss.item()))
                    continue

                r_pred = self.dis(high_res_images.to(self.device)).detach()
                f_pred = self.dis(fake_images)

                loss_g = self.loss_func_logits(f_pred - r_pred.mean(0, keepdim=True), real)

                fake_features = self.ext(fake_images)
                real_features = self.ext(high_res_images).detach()
                content_loss = self.content_loss_func(fake_features, real_features)

                g_loss = content_loss + self.lambda_adv*loss_g + self.lambda_pixel*p_loss

                g_loss.backward()
                self.optimizer_g.step()

                # train discriminator

                self.optimizer_d.zero_grad()

                r_pred = self.dis(high_res_images)
                f_pred = self.dis(fake_images.detach())

                r_loss = self.loss_func_logits(r_pred - f_pred.mean(0, keepdim=True), real)
                f_loss = self.loss_func_logits(f_pred - r_pred.mean(0, keepdim=True), fake)

                d_loss = (r_loss + f_loss) / 2

                d_loss.backward()
                self.optimizer_d.step()

                if i % 10 == 0:
                    print('     [{}/{}][{}/{}],  Gen Loss: {:.4f},   Dis Loss: {:.4f}\n'.format(epoch, epochs, i, len(self.train_loader), g_loss.item()/b_size, d_loss.item()/b_size))
                    gen_loss_list.append(g_loss.item()/b_size)
                    dis_loss_list.append(d_loss.item()/b_size)

            cur_time = time.time() - cur_time

            print('Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining\n'.format(cur_time, (epochs-epoch)*(cur_time)/3600))

            # show samples
            low_res_sample, high_res_sample = next(iter(self.test_loader))
            idx = np.random.randint(0, self.batch_size, 1)
            fake_image = self.gen(low_res_sample[idx].to(self.device))
            fake_image = fake_image.cpu().detach()
            ground_truth = high_res_sample[idx]
            image_grid = torchvision.utils.make_grid([fake_image[0], ground_truth[0]], nrow=2, normalize=True)
            _, plot = plt.subplots(figsize=(12, 12))
            plt.axis('off')
            plot.imshow(image_grid.permute(1, 2, 0))
            plt.savefig(saved_image_directory + '/epoch_{}_checkpoint.jpg'.format(epoch), bbox_inches='tight')

            # save models to model_directory
            torch.save(self.gen.state_dict(), saved_model_directory + '/generator_{}.pt'.format(epoch))
            torch.save(self.dis.state_dict(), saved_model_directory + '/discriminator_{}.pt'.format(epoch))
            torch.save(self.optimizer_g.state_dict(), saved_model_directory + '/optimizer_g_{}.pt'.format(epoch))
            torch.save(self.optimizer_d.state_dict(), saved_model_directory + '/optimizer_d_{}.pt'.format(epoch))


        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
        return gen_loss_list, dis_loss_list


def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')

    # hyperparameter loading
    parser.add_argument('--lr_dir_train', type=str,default='data/DIV2K_train_LR_mild', help='directory to low resolution training set')
    parser.add_argument('--hr_dir_train', type=str,default='data/DIV2K_train_HR', help='directory to high resolution training set')
    parser.add_argument('--lr_dir_test', type=str,default='data/DIV2K_valid_LR_mild', help='directory to low resolution test set')
    parser.add_argument('--hr_dir_test', type=str,default='data/DIV2K_valid_HR', help='directory to high resolution test set')
    parser.add_argument('--lr_length', type=int,default=128, help='Length of low resolution image (square image ideal)')
    parser.add_argument('--hr_length', type=int,default=512, help='Length of high resolution image (square image ideal)')
    parser.add_argument('--saved_image_directory', type=str, default='data/saved_images', help='directory to where image samples will be saved')
    parser.add_argument('--saved_model_directory', type=str, default='saved_models', help='directory to where model weights will be saved')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batches passed through networks at each step')
    parser.add_argument('--lambda_adv', type=float, default=5e-3, help='lambda factor for gan loss')
    parser.add_argument('--lambda_pixel', type=float, default=1e-2, help='lambda factor for generator pixel loss')
    parser.add_argument('--b1', type=float, default=0.9, help='optimizer beta 1 factor')
    parser.add_argument('--b2', type=float, default=0.999, help='optimizer beta 2 factor')
    parser.add_argument('--channels', type=int, default=3, help='number of color channels in images')
    parser.add_argument('--feature_space', type=int, default=64, help='ideal feature space for models to work in')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu depending on availability and compatability')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of models')
    parser.add_argument('--num_workers', type=int, default=0, help='workers simultaneously putting data into RAM')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations of dataset through network for training')
    args = parser.parse_args()

    lr_dir_train = args.lr_dir_train
    hr_dir_train = args.hr_dir_train
    lr_dir_test = args.lr_dir_test
    hr_dir_test = args.hr_dir_test
    lr_length = args.lr_length
    hr_length = args.hr_length 
    saved_image_dir = args.saved_image_directory
    saved_model_dir = args.saved_model_directory
    batch_size = args.batch_size
    lambda_adv = args.lambda_adv
    lambda_pixel = args.lambda_pixel
    b1 = args.b1 
    b2 = args.b2
    channels = args.channels
    feature_space = args.feature_space
    device = args.device
    lr = args.lr
    num_workers = args.num_workers
    epochs = args.epochs

    gan = Trainer(lr_dir_train, hr_dir_train, lr_dir_test, hr_dir_test, lr_length, hr_length, batch_size, lambda_adv, lambda_pixel, b1, b2, channels, feature_space, device, lr, num_workers)
    gen_loss_lost, dis_loss_list = gan.train(
        epochs, saved_image_dir, saved_model_dir)


if __name__ == "__main__":
    main()
