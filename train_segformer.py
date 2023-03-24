import sys
sys.path.append('.')
sys.path.append('Models')
from RangeViT import *
from utils import *

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

"""#Model"""

#!pip install transformers

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

import torch.nn as nn
model.segformer.encoder.patch_embeddings[0].proj=nn.Conv2d(5, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
model.decode_head.classifier = nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))

"""#Training"""

train_set = RangeKitti('range_dataset', mode='train')
val_set = RangeKitti('range_dataset', mode='test')

train_loader = data.DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=2)

val_loader = data.DataLoader(
        val_set,
        batch_size=2,
        shuffle=True,
        num_workers=2)

model.cuda()

##

# here are the training parameters
batch_size = 32
learning_rate =1e-3
weight_decay=2e-4
lr_decay_epochs=20
lr_decay=0.1
nb_epochs=25

# We are going to use the CrossEntropyLoss loss function as it's most
# frequentely used in classification problems with multiple classes which
# fits the problem. This criterion  combines LogSoftMax and NLLLoss.
criterion = nn.CrossEntropyLoss(weight=weights.cuda().float())

# We build the optimizer
optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

# Learning rate decay scheduler
lr_updater = lr_scheduler.StepLR(optimizer, lr_decay_epochs,
                                     lr_decay)

# Evaluation metric
ignore_index=[]
#ignore_index0 = list(class_encoding).index('unlabeled')
ignore_index.append(0)
metric = IoU(20, ignore_index=ignore_index)

# Start Training
best_miou = 0
train_loss_history_1 = []
val_loss_history_1 = []
train_miou_history_1 = []
val_miou_history_1 = []
for epoch in range( nb_epochs):
  print(">>>> [Epoch: {0:d}] Training".format(epoch))
  
  epoch_loss, (iou, miou) = train_for_segformer(model, train_loader, optimizer, criterion, metric) 
  lr_updater.step()
  print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, miou))
  train_miou=miou
  train_loss=epoch_loss
  if (epoch + 1) % 5 == 0 or epoch + 1 == nb_epochs:

    print(">>>> [Epoch: {0:d}] Validation".format(epoch))
    loss, (iou, miou) = test_for_segformer(model, val_loader, criterion, metric)
    print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, loss, miou))
    train_loss_history_1.append(train_loss)
    val_loss_history_1.append(loss)
    train_miou_history_1.append(train_miou)
    val_miou_history_1.append(miou)
    # Print per class IoU on last epoch or if best iou
    if epoch + 1 == nb_epochs or miou > best_miou:
      for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))
        # Save the model if it's the best thus far
        if miou > best_miou:
          print("\nBest model thus far. Saving...\n")
          best_miou = miou
          torch.save(model.state_dict(), "TrainedModels/segformer_epoch{}.pt".format(epoch+1))


torch.save(model.state_dict(), "TrainedModels/segformer_epoch{}_final.pt".format(nb_epochs))

print('train_loss_history', train_loss_history_1)
print('val_loss_history',val_loss_history_1)
print('train_miou_history',train_miou_history_1)
print('val_miou_history',val_miou_history_1)
