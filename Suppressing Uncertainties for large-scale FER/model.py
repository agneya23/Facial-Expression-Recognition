import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

# CNN Backbone
class feat_extract(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet18(pretrained=True)
        self.cnn_backbone = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(img):
        features =  self.cnn_backbone(img)
        return features

# Self-Attention Importance Weighing Module
class self_attn_mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Sequential(
                        torch.nn.Linear(),
                        torch.nn.Sigmoid()
                        )

    def forward(self, features):
        alpha_lst = self.self_attn(features)
        return alpha_lst

# Rank Regularization Module
class rank_reg_mod():
    def __init__(self):
        pass

    def Loss(self, alpha_lst):
        alpha_lst = alpha_lst.sort(reverse=True)
        num = len(alpha_lst)
        high = mean(alpha_lst[:0.7*num])
        low = mean(alpha_lst[0.7*num:])
        rr_loss = max(0, 0.15 - (high - low))
        return rr_loss

# Relabeling Module
class relabeling_mod():
    def __init__():
        pass

# Final Classification Module
class classification_mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classification = torch.nn.Sequential(
                            torch.nn.Linear(),
                            torch.special.logit(),
                            torch.nn.Softmax()
                            )

    def forward(self, features):
        output = self.classification(features)
        return output
