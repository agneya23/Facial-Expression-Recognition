import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def train(cnn_backbone, self_attn, rank_reg, classification, ce_loss, loader):
    for img, label in train_loader:
        optimizer.zero_grad()

        features = feat_extract.cnn_backbone(img)
        alpha_lst = self_attn_mod.self_attn(features)
        rr_loss = rank_reg_mod.rank_reg(alpha_lst)
        ce_loss = classification_mod.classification(features)

        total_loss = rr_loss + ce_loss
        total_loss.backward(retain_graph=False)
        optimizer.step()
