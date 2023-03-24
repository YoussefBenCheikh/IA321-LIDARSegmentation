
import sys
sys.path.append('.')
sys.path.append('Models')
from RangeViT import *
from utils import *
from UNet import *

import random
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image





PATH_TO_MODEL = sys.argv[1]
        
model = UNet(classes=20)

device = torch.device("cuda")
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.to(device)






val_set = RangeKitti('range_dataset', mode='test')



val_loader = data.DataLoader(
        val_set,
        batch_size=2,
        shuffle=True,
        num_workers=2)

model.cuda()

##

# here are the training parameters
batch_size = 32

criterion = nn.CrossEntropyLoss(weight=weights.cuda().float())
# We are going to use the CrossEntropyLoss loss function as it's most
# frequentely used in classification problems with multiple classes which
# fits the problem. This criterion  combines LogSoftMax and NLLLoss.


# Evaluation metric
ignore_index=[]
#ignore_index0 = list(class_encoding).index('unlabeled')
ignore_index.append(0)
metric = IoU(20, ignore_index=ignore_index)

index = int(sys.argv[2])
example, target = val_set[index]

prd = model(example.cuda().unsqueeze(0))

prediction = torch.argmax(prd[0], dim=0)
print(prediction)
plt.imsave("pred_unet{}.png".format(index), prediction.cpu().numpy().astype(np.float))
plt.imsave("gt{}.png".format(index), target)
    
    

