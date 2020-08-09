import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import imageio
import torch.nn as nn
import torch.nn.functional as F
import torch 
import timeit
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from read_data import read_file, FixationDataset, get_file_names_test
from network import Encoder, Decoder, Generator


class Rescale_test():
    def __init__(self):
        pass
    def __call__(self, sample):
        self.image = sample.astype(np.float32) / 255.0
        
        return self.image
    
class ToTensor_test():
    def __init__(self):
        pass
    def __call__(self, sample):
        self.image = sample.T
        self.image = torch.from_numpy(self.image)
        return self.image

class TestDataset(Dataset):
    def __init__(self, root_dir, image_file, transform=None):
        self.root_dir = root_dir
        self.image_files = read_file(image_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        if self.transform:
            image = self.transform(image)
            
        return image

def save_image(image, path, image_name):
    os.chdir(path)
    imageio.imwrite(image_name, image)
    print(image_name + " - saved")


def show(img, pred_img, path, file_name):
    pilTrans = transforms.ToPILImage()
    pilImg_org = pilTrans(img.permute(0,2,1))
    pilImg_pred = pilTrans(pred_img.permute(0,2,1))
    img_org = np.array(pilImg_org)
    img_pred = np.array(pilImg_pred)

    ## SHOW IMAGES
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img_org)
    ax2.imshow(img_pred, cmap='gray')
    plt.show()

    ## SAVE IMAGE
    # save_image(img_pred, path, file_name)
    
def predict(model, img, path, file_name):
    img_org = img
    # show(img.cpu().data.squeeze(0), gray=False)
    if USE_GPU == True:
        out = model(img.cuda())
    else:
        out = model(img)
    out = torch.sigmoid(out)

    out = nn.Upsample(scale_factor=8, mode='bilinear', 
                                    align_corners=False)(out)
    pred_img = out.cpu().data.squeeze(0)
    
    # print(pred_img.shape)
    show(img_org.cpu().data.squeeze(0), pred_img, path, file_name)

root_dir_test = '' ##PATH TO TEST IMAGE DIRECTORY
test_images = root_dir_test + 'test_images.txt'
DATA_FOLDER = '' ##MAIN DIRECTORY PATH
file_names = get_file_names_test(test_images)
USE_GPU = True

test_transform = torchvision.transforms.Compose([Rescale_test(), ToTensor_test()])
test_dl = DataLoader(TestDataset(root_dir_test, test_images, transform=test_transform), batch_size=1)

MODEL_PATH = DATA_FOLDER + '\\eye_fixation_weights.pth'
if os.path.exists(MODEL_PATH):
    trained_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    print("model loaded")

model = Generator()
model.load_state_dict(trained_model['model_state_dict'])
model.eval()

if torch.cuda.is_available():
  model.cuda()
  USE_GPU = True
else:
    model.cpu()
    USE_GPU = False


DIR_TO_SAVE = root_dir_test + 'predicted_images_130epoch_model'
if not os.path.exists(DIR_TO_SAVE):
    os.mkdir(DIR_TO_SAVE)


itr = iter(test_dl)
for i in range(len(test_dl)):
    validation_sample = next(itr)
    predict(model, validation_sample, DIR_TO_SAVE, file_name=file_names[i])
    

