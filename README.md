# Image-Enhancer-via-ESRGAN
The utilities developed in this tool are based off of the ESRGAN Paper: https://arxiv.org/abs/1809.00219

This tool enhances image resolution quality using deep convolutional neural networks. The idea is to train two neural networks to work against each other (the fundamental principle of Generative Adversarial Networks or GAN for short). A Generator is first trained a specified number iterations over the ground truth pixel loss using Mean Absolute Error or L1Loss. After the period of pixel loss is finished, the GAN is also tested on content loss (perceptual similarity) as well as the minimax GAN loss. The Discriminator is also introduced being trained on GAN loss. 

## Sample Low Resolution Images

![](data/uploads/low_res_samples.png)
![](data/uploads/low_res_samples2.png)

## GAN Enhanced Super Resolution Samples

![](data/uploads/gan_improved_samples_brightened.png)
![](data/uploads/gan_improved_sample2_brightened.png)

## One-to-One Comparions (Low Res --> Enhanced)

### 150 Epochs Trained
![](data/uploads/121_lr.png) ![](data/uploads/121_enhanced.png)

### 50 Epochs Trained
![](data/uploads/low_res_5.png) ![](data/uploads/enhanced_image_5.png)

### 25 Epochs Trained
![](data/uploads/low_res_4.png) ![](data/uploads/enhanced_image_4.png)
