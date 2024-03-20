import numpy as np 
import cv2
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def normalize_image(image):
    return image/255
def cropping(image, image2):
    original_size = (image.size()[-2], image.size()[-1])
    need_size = (image2.size()[-2], image2.size()[-1])
    image = image[:,:,:int(need_size[0] + (original_size[0] - need_size[0])/2), :int(need_size[1] + (original_size[1] - need_size[1])/2)]
    image = image[:,:,int((original_size[0] - need_size[0])/2):,int((original_size[1] - need_size[1])/2):]
    return image
class ModelCheckpoint(nn.Module):
    def __init__(self, directory):
        super(ModelCheckpoint, self).__init__()
        self.directory = directory

        # Create the directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)

    def forward(self, model, epoch):
        # Generate filename based on the epoch number
        filepath = os.path.join(self.directory, f"model_weights_epoch_{epoch}.pt")
        # Save the model weights to the generated file
        torch.save(model.state_dict(), filepath)
        print(f"Saved model weights at epoch {epoch} to {filepath}.")
class covidDataset(Dataset): 
    def __init__(self, imgs, masks):
        self.imgs_link = imgs
        self.masks_link = masks
        self.imgs = sorted([x for x in os.listdir(imgs)])
        self.masks = sorted([x for x in os.listdir(masks)])
    def __len__(self):
        return len((self.imgs))
    def __getitem__(self, index):
        image_path = os.path.join(self.imgs_link, self.imgs[index])
        mask_path = os.path.join(self.masks_link, self.masks[index])
        image = np.array(cv2.imread(image_path, flags = cv2.IMREAD_GRAYSCALE))
        image = torch.FloatTensor(normalize_image(image))
        maskk = np.zeros((260, 260))
        mask = normalize_image((cv2.imread(mask_path,flags = cv2.IMREAD_GRAYSCALE)))
        maskk[2:-2,2:-2] = mask
        maskk = torch.FloatTensor(maskk)
        return image.unsqueeze(0), maskk.unsqueeze(0)
traindata = covidDataset("/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data/Train/COVID-19/images", "/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data/Train/COVID-19/infection masks")
testdata =  covidDataset("/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data/Val/COVID-19/images", "/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data/Val/COVID-19/infection masks")
testloader = DataLoader(testdata, shuffle = False)
trainloader = DataLoader(traindata, shuffle = True)
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = (94,94)),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )
        self.in2 = nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU()
        )
        self.in3 = nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU()
        )
        self.in4 = nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU()
        )
        self.in5 = nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, 2)
        )
        self.out1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2)
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2)
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2)
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1), 
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, image):
        output1 = self.in1(image)
        output2 = self.in2(output1)
        output3 = self.in3(output2)
        output4 = self.in4(output3)
        output5 = self.in5(output4)
        output6 = self.out1(torch.cat([output5, cropping(output4, output5)],  dim = 1))
        output7 = self.out2(torch.cat([output6, cropping(output3, output6)], dim = 1))
        output8 = self.out3(torch.cat([output7, cropping(output2, output7)], dim = 1))
        output9 = self.out4(torch.cat([output8, cropping(output1, output8)], dim = 1))
        return output9
class ModelCheckpoint(nn.Module):
    def __init__(self, directory):
        super(ModelCheckpoint, self).__init__()
        self.directory = directory

        # Create the directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)

    def forward(self, model, epoch):
        # Generate filename based on the epoch number
        filepath = os.path.join(self.directory, f"model_weights_epoch_{epoch}.pt")
        # Save the model weights to the generated file
        torch.save(model.state_dict(), filepath)
        print(f"Saved model weights at epoch {epoch} to {filepath}.")
model = UNet().to("cuda")
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0001)
criterion = nn.BCELoss()
checkpoint_callback = ModelCheckpoint('/kaggle/working/model_weights')
from tqdm import tqdm
n_epochs = 50
train_loss = []
val_loss = []
for epoch in (range(n_epochs)):
    total_loss = 0.0
    model.train()
    for index, data in tqdm(enumerate(trainloader)):
        x, y = data[0].to("cuda"), data[1].to("cuda")
        y_pred = model(x)
        loss = criterion(y_pred, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train loss: {0}, Acc:'.format(total_loss/len(trainloader)))
    train_loss.append(total_loss/len(trainloader))
    if(epoch % 10 == 0):
        checkpoint_callback(model, epoch)
    model.eval()
    running_loss = 0.0

    for batch_index, batch in enumerate(testloader):
        x_batch, y_batch = batch[0].to("cuda"), batch[1].to("cuda")

        with torch.no_grad():
            output = model(x_batch)
            loss = criterion(output, y_batch)
            running_loss += loss.item()
    avg_loss_across_batches = running_loss / len(testloader)
    val_loss.append(avg_loss_across_batches)
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))