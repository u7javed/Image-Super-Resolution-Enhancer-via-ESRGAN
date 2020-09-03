import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = torchvision.models.vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)

class DenseBlock(nn.Module):
    def __init__(self, conv_in, conv_out, k_size, beta=0.2):
        super(DenseBlock, self).__init__()

        self.res1 = nn.Sequential(
            nn.Conv2d(conv_in, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(conv_in*2, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(conv_in*3, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.res4 = nn.Sequential(
            nn.Conv2d(conv_in*4, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.res5 = nn.Sequential(
            nn.Conv2d(conv_in*5, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1)
        )
        self.beta = beta

    def forward(self, input):
        x = input
        #feature size = convin*2
        result = self.res1(x)
        x = torch.cat([x, result], 1)

        result = self.res2(x)
        x = torch.cat([x, result], 1)

        result = self.res3(x)
        x = torch.cat([x, result], 1)

        result = self.res4(x)
        x = torch.cat([x, result], 1)

        x = self.res5(x)

        output = x.mul(self.beta)
        return output + input

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, conv_in, k_size, beta=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()

        self.residual = DenseBlock(conv_in, conv_in, k_size)
        self.beta = beta

    def forward(self, input):
        x = self.residual(input)
        x = self.residual(x)
        x = self.residual(x)
        output = x.mul(self.beta)
        return output + input


class Generator(nn.Module):
    def __init__(self, channels, feature_space=64, n_rrsd=16, upsample_2div=2, beta=0.2):
        super(Generator, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(channels, feature_space, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )

        #number of rrsd layers
        rrsd_layers = []
        for _ in range(n_rrsd):
            rrsd_layers.append(ResidualInResidualDenseBlock(feature_space, 3, beta))

        self.rrsd_sequence = nn.Sequential(*rrsd_layers)

        #second conv
        self.second_conv = nn.Conv2d(feature_space, feature_space, kernel_size=(3, 3), stride=1, padding=1)

        #upsampling
        upsampling_layers = []
        for _ in range(upsample_2div):
            upsampling_layers.append(nn.Conv2d(feature_space, feature_space*4, kernel_size=(3, 3), stride=1, padding=1))
            upsampling_layers.append(nn.LeakyReLU())
            upsampling_layers.append(nn.PixelShuffle(upscale_factor=2))

        self.upsample_sequence = nn.Sequential(*upsampling_layers)

        #final conv
        self.final_conv = nn.Conv2d(feature_space, channels, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, input):
        x = input
        conv1 = self.init_conv(x)
        rrsd_output = self.rrsd_sequence(conv1)
        conv2 = self.second_conv(rrsd_output)
        output = torch.add(conv1, conv2)
        output = self.upsample_sequence(output)
        return self.final_conv(output)


class Discriminator(nn.Module):
    def DiscriminatorBlock(self, conv_in, conv_out, k_size, stride):
        architecture = []
        architecture.append(nn.Conv2d(conv_in, conv_out, kernel_size=(k_size, k_size), stride=stride, padding=1))
        architecture.append(nn.BatchNorm2d(conv_out))
        architecture.append(nn.LeakyReLU(0.2))
        return architecture

    def __init__(self, input_length, channels, init_feature_space=64):
        super(Discriminator, self).__init__()

        self.input_size = input_length

        self.init_conv = nn.Sequential(
            nn.Conv2d(channels, init_feature_space, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        downsize_layers = []
        downsize_layers.extend(self.DiscriminatorBlock(init_feature_space, init_feature_space*2, k_size=3, stride=2))
        downsize_layers.extend(self.DiscriminatorBlock(init_feature_space*2, init_feature_space*2, k_size=3, stride=1))
        downsize_layers.extend(self.DiscriminatorBlock(init_feature_space*2, init_feature_space*4, k_size=3, stride=2))
        downsize_layers.extend(self.DiscriminatorBlock(init_feature_space*4, init_feature_space*4, k_size=3, stride=1))
        downsize_layers.extend(self.DiscriminatorBlock(init_feature_space*4, init_feature_space*8, k_size=3, stride=2))
        downsize_layers.extend(self.DiscriminatorBlock(init_feature_space*8, init_feature_space*8, k_size=3, stride=1))
        downsize_layers.extend(self.DiscriminatorBlock(init_feature_space*8, init_feature_space*8, k_size=3, stride=2))

        self.downsize_sequence = nn.Sequential(*downsize_layers)
        self.output_shape = (1, self.input_size//16, self.input_size//16)

        self.final_conv = nn.Conv2d(init_feature_space*8, 1, kernel_size=(3, 3), stride=1, padding=1)


    def forward(self, input):
        x = self.init_conv(input)
        x = self.downsize_sequence(x)
        x = self.final_conv(x)
        return x
        



