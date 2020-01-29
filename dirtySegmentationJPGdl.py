import pandas as pd
import os
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool



#download the below csv for the training annotations for all of the images with segmentation data
# https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv
#move or copy it into a new folder name "oiv5segmentation" in your home directory along with this script
f=pd.read_csv("/home/fjsaezm/oiv5segmentation/train-annotations-object-segmentation.csv")




#download the "class-descriptions-boxable.csv" from the OIDV5 downloads page here: 
# https://storage.googleapis.com/openimages/web/download.html  
#you can find a list of LabelNames for classes that have segmentation data here:
# https://storage.googleapis.com/openimages/v5/classes-segmentation.txt
# Classes to be used
# /m/01g317	Person
# /m/015qff Traffic Light
# /m/0k4j	Car
# /m/01mqdt	Traffic sign


#classNames = ["Person","TrafficLight","Car","TrafficSign"]
#numClasses = ['01g317','015qff','0k4j','01mqdt']

print("Ready to load imgs")
name_num = [["Person",['/m/01g317']],["TrafficLight",['/m/015qff']],["Car",['/m/0k4j']],["TrafficSign",['/m/01mqdt']]]

threads = 20
nImages = 3000
commands = []

os.mkdir("images3")

download_dir = "/home/fjsaezm/oiv5segmentation/images3/"

for el in name_num:
    print(el[0])
    u = f.loc[f['LabelName'].isin(el[1])]
    print(u)
    pool = ThreadPool(threads)

    for ind in u.index[0:nImages]:
        image = u['ImageID'][ind]


        # Train images
        path = "train" + '/' + str(image) + '.jpg ' + '"' + download_dir + '"'
        command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/' + path
        if not command in commands:
            commands.append(command)

    #Reset current dir
    print("Added commands for " + el[0])
    os.chdir("..")

print(commands)
list(tqdm(pool.imap(os.system, commands), total = len(commands)))


print('Done!')
pool.close()
pool.join()

#YOU STILL NEED TO DOWNLOAD THE MASK PNG's FROM THE DOWNLOADS PAGE!! 
#go here:
#  https://storage.googleapis.com/openimages/web/download.html
#click "train" by "segmentations" and download all of the zip files (0 1 2 3 4 5 6 7 8 9 a b c d e and f)
#It's not that big of a set of files, just download all of them by clicking on the links
