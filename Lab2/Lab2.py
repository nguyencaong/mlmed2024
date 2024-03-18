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
from tqdm import tqdm  # Import tqdm
def masking(img):
    # im_gr = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = np.array(img, np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])
    return cv2.ellipse(img, ellipse, (255,255,255), -1)
def toString(number):
    if number < 10: 
        return "00" + str(number)
    elif 10 <= number <100: 
        return "0" + str(number)
    else: 
        return str(number)
def normalize_image(image):
    return image/255

def croppingandpadding(image):
    #cropping
    if image.shape[1] > 800:
        image = image[:,:800]
    if image.shape[0] > 540:
        image = image[:540,:]
    if image.shape[1] < 800 or image.shape[0] < 540:
        mask = np.zeros((540, 800))
        mask[:image.shape[0], :image.shape[1]] = image
        image = mask
    return image
def cropping(image, image2):
    original_size = (image.size()[-2], image.size()[-1])
    need_size = (image2.size()[-2], image2.size()[-1])
    image = image[:,:,:int(need_size[0] + (original_size[0] - need_size[0])/2), :int(need_size[1] + (original_size[1] - need_size[1])/2)]
    image = image[:,:,int((original_size[0] - need_size[0])/2):,int((original_size[1] - need_size[1])/2):]
    return image
class ModelCheckpoint(nn.Module):
    def __init__(self, filepath, save_freq):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq
    
    def forward(self, epoch):
        if epoch % self.save_freq == 0:
            torch.save(self.state_dict(), self.filepath.format(epoch=epoch))
class HCDataset(Dataset): 
    def __init__(self, ref):
        self.ref = ref
        self.imgs = sorted([x for x in os.listdir(ref) if "Annotation" not in x])
        self.masks = sorted([x for x in os.listdir(ref) if "Annotation" in x])
    def __len__(self):
        return len(sorted([x for x in os.listdir(self.ref) if "Annotation" not in x]))
    def __getitem__(self, index):
        image_path = os.path.join(self.ref, self.imgs[index])
        mask_path = os.path.join(self.ref, self.masks[index])
        image = np.array(cv2.imread(image_path, flags = cv2.IMREAD_GRAYSCALE))
        image = normalize_image(image)
        image = torch.FloatTensor(croppingandpadding(image))
        maskk = np.zeros((548, 804))
        mask = croppingandpadding(normalize_image(masking(cv2.imread(mask_path,flags = cv2.IMREAD_GRAYSCALE))))
        maskk[4:-4,2:-2] = mask
        maskk = torch.FloatTensor(maskk)
        return image.unsqueeze(0), maskk.unsqueeze(0)
traindata = HCDataset("/kaggle/input/fetalllllll/training_set/training_set")
trainloader = DataLoader(traindata, shuffle = True)
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
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = (96,94)),
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

model = UNet().to("cuda")
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0001)
criterion = nn.BCELoss()
checkpoint_callback = ModelCheckpoint(directory='/kaggle/working/model_weights')
n_epochs = 35
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
#     correct = 0
#     with torch.no_grad():
#         y_train_pred = model(torch.Tensor(X_train).unsqueeze(1))
#         for i in range(len(y_train_pred)):
#             if np.argmax(y_train_pred[i]) == int(y_train[i]):
#                 correct += 1
#     test_accuracy = correct / len(y_train)
    print('Train loss: {0}, Acc:'.format(total_loss/len(trainloader)))
    train_loss.append(total_loss/len(trainloader))
    if(epoch % 10 == 0):
        checkpoint_callback(model, epoch)
#     model.eval()
#     running_loss = 0.0

#     for batch_index, batch in enumerate(valid_loader):
#         x_batch, y_batch = batch[0], batch[1]

#         with torch.no_grad():
#             output = model(x_batch)
#             loss = criterion(output, y_batch.squeeze())
#             running_loss += loss.item()
#     avg_loss_across_batches = running_loss / len(valid_loader)
#     correct = 0
#     with torch.no_grad():
#         y_valid_pred = model(torch.Tensor(X_test).unsqueeze(1))
#         for i in range(len(y_valid_pred)):
#             if np.argmax(y_valid_pred[i]) == int(y_test[i]):
#                 correct += 1
#     accuracy = correct / len(y_test)
#     val_loss.append(avg_loss_across_batches)
#     print('Val Loss: {0:.3f}, Acc: {1}'.format(avg_loss_across_batches, accuracy))
test = np.array(cv2.imread("/kaggle/input/fetalllllll/test_set/test_set/000_HC.png", flags = cv2.IMREAD_GRAYSCALE))
test = normalize_image(test)
test = torch.FloatTensor(croppingandpadding(test))
import matplotlib.pyplot as plt
plt.imshow(test) 
plt.imshow(model((test.unsqueeze(0)).unsqueeze(0).to("cuda")).detach().cpu().numpy()[0][0])