from torch.utils.data import Dataset
import torch
import os
import imageio

def read_file(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file: 
            line = line.strip() 
            lines.append(line)
    return lines


class FixationDataset(Dataset):
    def __init__(self, root_dir, image_file, fixation_file, transform=None):
        self.root_dir = root_dir
        self.image_files = read_file(image_file)
        self.fixation_files = read_file(fixation_file)
        self.transform = transform
        assert(len(self.image_files) == len(self.fixation_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
        fix = imageio.imread(fix_name)

        sample = {'image': image, 'fixation': fix}
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def get_file_names_test(test_images):
    names = read_file(test_images)
    trunc_names = []
    for i in names:
        striped_name = i.split('-')
        trunc_names.append("prediction-"+striped_name[1])
    
    return trunc_names