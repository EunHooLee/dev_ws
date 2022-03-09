import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataset




sample_dir = 'samples_fashion'

if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

EPOCHS = 500
BATCH_SIZE = 100
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

trainset = datasets.FashionMNIST('./data',
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.5,),(0.5,))
                                 ]))

train_loader = torch.utils.data.DataLoader(
    dataset=trainset, batch_size=BATCH_SIZE, shuffle=True
)

G = nn.Sequential(
    nn.Linear(64,256),
    nn.ReLU(),
    nn.Linear(256,256),
    nn.ReLU(),
    nn.Linear(256,784),
    nn.Tanh()
)

D = nn.Sequential(
    nn.Linear(784,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,1),
    nn.Sigmoid()
)

D = D.to(device)
G = G.to(device)

criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)


total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images,_) in enumerate(train_loader): # 100 개씩 총 600번 반복 (60000개 데이터)
        images = images.reshape(BATCH_SIZE,-1).to(device) # 100 x 784

        real_labels = torch.ones(BATCH_SIZE,1).to(device)  # 100 x 1
        fake_labels = torch.zeros(BATCH_SIZE,1).to(device) # 100 x 1

        outputs = D(images) # 100 x 784 이미지를 784 input 을 갖은 D 로 넣는다.
        d_loss_real = criterion(outputs,real_labels)  # 진짜 이미지를 줬을 때 거짓이라고 하는 경우
        real_score = outputs

        z =torch.randn(BATCH_SIZE,64).to(device) # latent vector 생성
        fake_images = G(z) # Fake image를 만든다.

        outputs = D(fake_images) # 생성된 가짜 이미지를 입력한다.
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # 최종 D의 오차는 진짜 이미지를 줬을 때 거짓이라고 인식한 경우와
        # Generator에서 생성된 이미지를 줬을 때 진짜라고 인식한 경우의 오차의 합이다.
        d_loss = d_loss_fake + d_loss_real

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        d_loss.backward()
        d_optimizer.step()

        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
    print('Epoch [{}/{}], d_loss: {:.4f} g_loss: {:.4f} D(x): {:.2f}, D(G(z)): {:.2f}'.format(epoch, EPOCHS, d_loss.item(),g_loss.item(),real_score.mean().item(), fake_score.mean().item()))

    if (epoch+1) == 1:
        images = images.reshape(images.size(0),1,28,28)
        torchvision.utils.save_image(images,os.path.join(sample_dir, 'real_images.png'))
    fake_images = fake_images.reshape(fake_images.size(0),1,28,28)
    save_image(fake_images,os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))


torch.save(G.state_dict(), 'G_fashion.ckpt')
torch.save(D.state_dict(), 'D_fashion.ckpt')
