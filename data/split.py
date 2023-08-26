# split solar image into train/val/test
import os
import numpy as np
import shutil
import random

# root_dir = 'data/allsolar'
# root_dir = 'data/allsolar_224'
root_dir = 'data/allsolar_png1500'
new_root = 'data'
check_extenstion=".png"
classes = ['pos', 'neg']
test_ratio = 0.20
Seed = 0
print("Random Seed: ", Seed)
random.seed(Seed)
np.random.seed(Seed)
for cls in classes:
    if not os.path.exists(os.path.join(new_root,'train',cls)):
        os.makedirs(os.path.join(new_root,'train',cls) )
    if not os.path.exists(os.path.join(new_root,'val',cls)):
        os.makedirs(os.path.join(new_root,'val',cls) )
    if not os.path.exists(os.path.join(new_root,'test',cls)):    
        os.makedirs(os.path.join(new_root,'test',cls) )
    
for cls in classes:
    src = os.path.join(root_dir,cls)  # folder to copy images from
    print(src)

    allFileNames = os.listdir(src)
    allFileNames=[ filename for filename in allFileNames if filename.endswith( check_extenstion) ]
    np.random.shuffle(allFileNames)

    val_test_split = int(np.around( test_ratio * len(allFileNames) ))
    train_val_split = int(len(allFileNames)-2*val_test_split)
    train_FileNames = allFileNames[:train_val_split]
    val_FileNames = allFileNames[train_val_split:train_val_split+val_test_split]
    test_FileNames = allFileNames[train_val_split+val_test_split:]   
                     
    # full path
    train_FileNames = [os.path.join(src,name) for name in train_FileNames]
    val_FileNames = [os.path.join(src,name) for name in val_FileNames]
    test_FileNames = [os.path.join(src,name) for name in test_FileNames]

    print('Total: ' +str(len(allFileNames)))
    print('Train: ' +str(len(train_FileNames)))
    print('Val  : ' +str(len(val_FileNames)))
    print('Test : ' +str(len(test_FileNames)))
    
    ## Copy pasting images to target directory
    for name in train_FileNames:
        shutil.copy(name,  os.path.join(new_root,'train',cls))
    for name in val_FileNames:
        shutil.copy(name, os.path.join(new_root,'val',cls))
    for name in test_FileNames:
        shutil.copy(name, os.path.join(new_root,'test',cls) )