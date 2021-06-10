import torch
import numpy as np

"""
Scene Detection and sort the distance between each  
"""

def Detect(latent, dis_metric, num_classes):
    n = latent.shape[0]
    dis_all = []
    for i in range(1, n):
        pre = latent[i-1, :]
        cur = latent[i, :]
        dis = dis_metric(pre, cur)
        dis_all.append(dis)
    dis_sorted = np.argsort(dis_all)
    return dis_sorted[:num_classes]




