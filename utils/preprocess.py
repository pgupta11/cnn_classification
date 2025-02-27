from torchvision import transforms
from tqdm.auto import tqdm
import torch

img_size = 32

def preprocess_data(dataset):
    inputs = []
    
    preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    ])
    for record in tqdm(dataset):
        image = record['img']
        label = record['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        inputs.append([preprocess(image), label])
 
    return inputs

def normalize_data(dataset):
    inputs = preprocess_data(dataset)
    mean = torch.stack([img for img, _ in inputs], dim=3).view(3, -1).mean(dim=1)
    std = torch.stack([img for img,_ in inputs],dim=3).view(3,-1).std(dim=1)
    print(mean, std)
    preprocess = transforms.Compose([transforms.Normalize(mean, std)])
    for i in tqdm(range(len(inputs))):
        inputs[i][0] = preprocess(inputs[i][0])
    return inputs