# WGAN-GP 实现

这是一个基于 PyTorch 的 Wasserstein Generative Adversarial Network with Gradient Penalty（WGAN-GP）实现。该模型是基于 WGAN 进行改进，通过引入梯度惩罚（GP）来稳定训练过程，避免了WGAN中常见的梯度爆炸或消失问题。你可以在自己的数据集上训练此网络，生成逼真的图像。

## 功能特点
- **梯度惩罚：** 通过引入梯度惩罚项，优化了判别器，使得模型训练更稳定。
- **数据预处理：** 支持自动裁剪或填充图像为正方形，或者保留长方形图像。
- **图像生成：** 训练过程中会定期生成并保存图像，帮助观察训练进度。
- **模型保存：** 每隔一定轮次保存生成器和判别器的模型。

## 其它
这是基于 https://github.com/bigBulen/DCGAN 的改进，所以其他参数基本没有变化，可以参考前者。

## 效果
- GPU：rtx3060 12g

迭代49次：
![samples_epoch49_iter23](https://github.com/user-attachments/assets/e4415e66-db82-4ae3-a05a-25198d815854)


