"""
输入图片必须为正方形，若想要保留长方形训练，可选择裁切or填补方案。
若想要保留整个图片，可将 加载数据集 的 CustomCrop()注释掉，反之则会裁剪成正方形
裁剪原则在CustomCrop类那儿，可以修改尺寸啥的
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
import time

import matplotlib.pyplot as plt
import numpy as np


# 超参数优化
class Options:
    def __init__(self):
        self.n_epochs = 500
        self.image_size = 128 #128
        self.batch_size = 64 #64
        self.dataset = "folder"
        self.dataroot = "train_images"  # 数据集路径
        self.nz = 256
        self.ngf = 128
        self.ndf = 128
        self.niter = 500  # 训练轮数
        self.lr = 0.00015
        self.cuda = True
        self.ngpu = 1
        self.netG = ""
        self.netD = ""
        self.outf = "results"
        self.manualSeed = None
        self.classes = None
        self.workers = 0
        self.n_critic = 4  # 判别器更新次数，过高的值会导致判别器过强，生成器无法学习
        # 优化参数
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.gp_weight = 8  # 梯度惩罚系数*（保守程度，越小越创新，但稳定性差）
        # 系统参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



opt = Options()


# 自定义填充函数：将图片填充为正方形
def pad_to_square(img, fill=0):
    """
    将图片填充为正方形：
    1. 如果图片是长方形，在短边两侧填充像素。
    2. 填充颜色默认黑色（fill=0），可改为其他颜色（如 fill=255 为白色）。
    """
    w, h = img.size
    if w == h:
        return img
    elif w > h:
        # 高度不足，填充上下
        padding = (0, (w - h) // 2, 0, (w - h) - (w - h) // 2)  # (左, 上, 右, 下)
    else:
        # 宽度不足，填充左右
        padding = ((h - w) // 2, 0, (h - w) - (h - w) // 2, 0)
    return transforms.functional.pad(img, padding, fill=fill)

''''''
# 设置随机种子
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 创建输出文件夹
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


class CustomCrop:
    def __call__(self, img):
        width, height = img.size
        # 计算裁剪高度（40% 的高度）
        crop_height = int(height * 0.35)
        # 计算正方形边长（不能超过原图的宽度）
        crop_size = min(crop_height, width)
        # 计算左侧和右侧，使裁剪区域居中
        left = (width - crop_size) // 2
        right = left + crop_size
        # 上方从 0 开始，下方是 crop_size
        top = 0
        bottom = crop_size
        return img.crop((left, top, right, bottom))


# 加载并增强数据集
transform = transforms.Compose([
    CustomCrop(),
    transforms.Resize(opt.image_size),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = dset.ImageFolder(
    root=opt.dataroot,
    transform=transform
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers)
)

# 改进的生成器结构（含残差连接）
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (nz, 1, 1)
            self._block(opt.nz, opt.ngf * 16, 4, 1, 0),  # 4x4
            self._block(opt.ngf * 16, opt.ngf * 8, 4, 2, 1),  # 8x8
            self._block(opt.ngf * 8, opt.ngf * 4, 4, 2, 1),  # 16x16
            self._block(opt.ngf * 4, opt.ngf * 2, 4, 2, 1),  # 32x32
            self._block(opt.ngf * 2, opt.ngf, 4, 2, 1),  # 64x64
            nn.ConvTranspose2d(opt.ngf, 3, 4, 2, 1),
            # nn.ConvTranspose2d(3, 3, 4, 2, 1),  # 256x256 额外的上采样层

            nn.Tanh()
        )

    def _block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
            ResidualBlock(out_ch)  # 添加残差连接
        )

    def forward(self, x):
        return self.main(x)


# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


# 改进的判别器结构（WGAN-GP专用）
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (3, 256, 256)
            self._block(3, opt.ndf, 4, 2, 1),  # 128x128
            self._block(opt.ndf, opt.ndf * 2, 4, 2, 1),  # 64x64
            self._block(opt.ndf * 2, opt.ndf * 4, 4, 2, 1),  # 32x32
            self._block(opt.ndf * 4, opt.ndf * 8, 4, 2, 1),  # 16x16
            self._block(opt.ndf * 8, opt.ndf * 16, 4, 2, 1),  # 8x8
            nn.Conv2d(opt.ndf * 16, 1, 4, 1, 0),  # 5x5
            # nn.Conv2d(opt.ndf * 4, 1, 4, 1, 0) # 原为 5 层，现减少到 3 层
        )

    def _block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.main(x).view(-1)


# 梯度惩罚计算
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(opt.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# 初始化模型
netG = Generator().to(opt.device)
netD = Discriminator().to(opt.device)

# 使用Adam优化器
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr * 1.2, betas=(opt.beta1, opt.beta2))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr * 0.8, betas=(opt.beta1, opt.beta2))


"""
# 数据预处理（优化版）
transform = transforms.Compose([
    transforms.Lambda(lambda img: pad_to_square(img)),
    transforms.Resize(opt.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
"""



# 调试
# 检查数据集大小
print(f"数据集大小: {len(dataset)}")
# 检查数据加载器批次数量
print(f"数据加载器批次数量: {len(dataloader)}")
# 检查批量大小
print(f"批量大小: {opt.batch_size}")
"""调试-绘图（拟合曲线）"""
# 添加实时显示图像的功能
def show_images(real_imgs, fake_imgs, epoch, iteration):
    # 使用matplotlib显示图片
    real_imgs = real_imgs.cpu().numpy()  # 转为numpy数组
    fake_imgs = fake_imgs.cpu().numpy()
    # 画图：真实图像 vs 生成图像
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.transpose(real_imgs[0], (1, 2, 0)), interpolation='nearest')  # 真实图像
    axs[0].set_title('Real Image')
    axs[1].imshow(np.transpose(fake_imgs[0], (1, 2, 0)), interpolation='nearest')  # 生成图像
    axs[1].set_title('Generated Image')
    for ax in axs:
        ax.axis('off')
    plt.suptitle(f"Epoch {epoch} Iteration {iteration}")
    plt.show()
    plt.pause(0.1)  # 暂停一下，确保图像更新

# 添加损失值实时显示的功能
def plot_losses(d_losses, g_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title(f"Losses at Epoch {epoch}")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.pause(0.1)  # 暂停一下，确保图像更新
# 在训练循环中进行调用
d_losses = []
g_losses = []

# -调试end-


# 训练循环
# 断点续传，继续迭代results.1
netG.load_state_dict(torch.load("results/netG_epoch64_1738782899.6213663.pth"))
netD.load_state_dict(torch.load("results/netD_epoch64_1738782901.3935187.pth"))
# 训练循环（for循环接断点续传）
for epoch in range(65, opt.n_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):

        real_imgs = real_imgs.to(opt.device)
        batch_size = real_imgs.size(0)

        # 训练判别器
        for _ in range(opt.n_critic):
            netD.zero_grad()

            # 真实样本
            real_validity = netD(real_imgs)

            # 生成样本
            z = torch.randn(batch_size, opt.nz, 1, 1).to(opt.device)
            fake_imgs = netG(z)
            fake_validity = netD(fake_imgs.detach())

            # 梯度惩罚
            gp = compute_gradient_penalty(netD, real_imgs, fake_imgs)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)


            # WGAN-GP损失
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.gp_weight * gp
            d_loss.backward(retain_graph=True)  # 在这里设置retain_graph=True
            optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        fake_validity = netD(fake_imgs)
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        optimizerG.step()

        """调试-绘图（拟合曲线）-每隔一定次数显示图像和损失"""
        # 保存损失值
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())


        if i % 100 == 0:
            # 显示实时生成的图像
            with torch.no_grad():
                fixed_z = torch.randn(16, opt.nz, 1, 1).to(opt.device)
                samples = netG(fixed_z)
                show_images(real_imgs, samples, epoch, i)

        print(f"[{epoch}/{opt.n_epochs}][{i}/{len(dataloader)}] "
              f"D: {d_loss.item():.4f} G: {g_loss.item():.4f} GP: {gp.item():.4f}")


        plot_losses(d_losses, g_losses, epoch)
        img, _ = dataset[0] #输出处理后图片
        plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5)  # 反归一化
        plt.show()
        """调试-绘图（拟合曲线）-每个epoch结束后显示损失变化图"""

        # 保存样本
        with torch.no_grad():
            fixed_z = torch.randn(16, opt.nz, 1, 1).to(opt.device)
            samples = netG(fixed_z)
            vutils.save_image(real_imgs, f"{opt.outf}/real_samples.png", normalize=True)
            vutils.save_image(samples, f"results/samples_epoch{epoch}_iter{i}_t-{time.time()}.png",
                              nrow=4, normalize=True)

    # 保存检查点
    # if epoch % 10 == 0:
    torch.save(netG.state_dict(), f"results/netG_epoch{epoch}_{time.time()}.pth")
    torch.save(netD.state_dict(), f"results/netD_epoch{epoch}_{time.time()}.pth")
