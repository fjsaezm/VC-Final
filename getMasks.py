from os.path import isfile, join
from os import listdir
import shutil
import pandas as pd
import os


imgs_path = "images3/"
# Read csv
f=pd.read_csv("/home/fjsaezm/oiv5segmentation/train-annotations-object-segmentation.csv")
# Read imgs from dir
files = [a for a in listdir(imgs_path) if isfile(join(imgs_path, a))]

# Copy 0*,1*,2* images onto 012images
i = 0
new_path = "012images/"
#os.mkdir("012images")
for f in files:
    if str(id)[0] in ['0','1','2']:
        i = i+1
        path = imgs_path + f
        shutil.copy(path,new_path)

print("Copied ",i, " images")

exit(1)

# Get img ids
onlyfiles = []
for a in files:
    id = a.split(".")[0]
    if id[0] == '0' or id[0] == '1' or id[0] == '2':
        onlyfiles.append(id)

# Restrict csv to img ids
new_csv = f.loc[f['ImageID'].isin(onlyfiles)]

target_dir = "masks3/"
os.mkdir("masks3")

i = 0
for img in onlyfiles:
    print(i)
    for index,row in new_csv.iterrows():
        if img == row['ImageID']:
            path = "../Downloads/train-masks-"+ img[0] +"/" +row['MaskPath']
            shutil.copy(path,target_dir)
            i = i+1

print(i)
