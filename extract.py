import pandas as pd
import os
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from os.path import isfile, join
from os import listdir
import shutil


os.mkdir("images012")
onlyfiles = [f for f in listdir("images3/") if isfile(join("images3/", f))]

target_dir = "images012/"

for a in onlyfiles:
    if a[0] == '0' or a[0] == '1' or a[0] == '2':
        path = "images3/"+a
        shutil.copy(path,target_dir)


exit(1)

sub = [f for f in onlyfiles if f[0] == '0' or f[0] == '1' or f[0] == '2']
onlyfiles = sub

masks0 = [f for f in listdir("../Downloads/train-masks-0/") if isfile(join("../Downloads/train-masks-0/",f))]

masks1 = [f for f in listdir("../Downloads/train-masks-1/") if isfile(join("../Downloads/train-masks-1/",f))]

masks2 = [f for f in listdir("../Downloads/train-masks-1/") if isfile(join("../Downloads/train-masks-1/",f))]


target_dir = "masks/"
#os.mkdir("masks")

for img in onlyfiles:
    id = img.split(".")[0]
    if id[0] == '0':
        for m in masks0:
            if id in m:
                path = "../Downloads/train-masks-0/"+m
                shutil.copy(path,target_dir)
    if id[0] == '1':
        for m in masks1:
            if id in m:
                path = "../Downloads/train-masks-1/"+m
                shutil.copy(path,target_dir)
    if id[0] == '2':
        for m in masks2:
            if id in m:
                path = "../Downloads/train-masks-2/"+m
                shutil.copy(path,target_dir)


