# change resolution of solar image
# modify file_extention, root_dir, new_root
import os
import numpy as np
import shutil
import random
import pandas as pd
import cv2
from PIL import Image

# file_extention='.csv'
# root_dir = 'data/allsolar'
# new_root = 'data/allsolar_224'
# classes = ['pos', 'neg']

# for cls in classes:
#     if not os.path.exists(os.path.join(new_root,cls)):
#         os.makedirs(os.path.join(new_root,cls) )
    
# for cls in classes:
#     src = os.path.join(root_dir,cls)  # folder to copy images from
#     dest=os.path.join(new_root,cls)
#     print(src)
#     allFileNames = os.listdir(src)
#     allFileNames=[ filename for filename in allFileNames if filename.endswith( file_extention ) ] 
#     for file in allFileNames:
#         if file_extention =='.csv':
#             data = pd.read_csv(os.path.join(src,file), header = None)
#             datanp=data.to_numpy()
#             datanp=cv2.resize(datanp, (224, 224))
#             np.savetxt(os.path.join(dest,file), datanp, delimiter=",")
            
            
file_extention='.png'
# root_dir = 'data/allsolar_png1500'
# new_root = 'data/allsolar_png512'

root_dir = 'data/allsolar_full_png'
new_root = 'data/allsolar_full_png512'
if not os.path.exists(os.path.join(new_root)):
    os.makedirs(os.path.join(new_root) )

src = os.path.join(root_dir)  # folder to copy images from
dest=os.path.join(new_root)
allFileNames = os.listdir(src)
allFileNames=[ filename for filename in allFileNames if filename.endswith( file_extention ) ] 
for file in allFileNames:
    img = Image.open(os.path.join(src,file),'r')
    data=(img.convert("L")) #greyscale
    data=data.resize( (512, 512))
    data.save(os.path.join(dest,file))           

