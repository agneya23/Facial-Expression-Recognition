import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def main_fer(dir_path):
    data_path = dir_path + 'fer2013.csv'
    train_lst, test_lst = load_images_fer(data_path)

    for img, emotion in train_lst:
        face_train = face_detect(img)
        x, y, w, h = face_train['box']
        img = img[x:x+w, y:y+h]
        img = torchvision.transforms.Resize([224, 224])(img)
    for img, emotion in test_lst:
        face_test = face_detect(img)
        x, y, w, h = face_test['box']
        img = img[x:x+w, y:y+h]
        img = torchvision.transforms.Resize([224, 224])(img)

    # Create Dataloaders
    train_loader = torch.utils.data.DataLoader(train_lst, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_lst, batch_size=1, shuffle=True)

    feat_extract = feat_extract()
    self_attn_mod = self_attn_mod()
    rank_reg_mod = rank_reg_mod()
    classification_mod = classification_mod()
    relabeling_mod = relabeling_mod()

    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params,weight_decay = 1e-4)

    epochs = 40
    for i in range(epochs):
        train(cnn_backbone, self_attn, rank_reg, classification, ce_loss, train_loader, optimizer)
