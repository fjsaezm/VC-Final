import os
from os.path import isfile,join
from os import listdir

i = 0

mask = "../../Downloads/train-masks-0/"
onlyfiles = [f for f in listdir(mask) if isfile(join(mask, f))]
id = "0a7e0b2c83069f3c"

for f in onlyfiles:
    if id in f:
        i = i+1

print(i)
