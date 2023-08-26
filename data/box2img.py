import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
""" 
iterate csv boxes in /box_4096 and convert them to images 
"""
# boxfiles_dir = 'data/box_4096'
# des_dir='data/allsolar_png1500_boximage'
boxfiles_dir = 'data/box_full_4096'
des_dir='data/allsolar_full_png512_boximage'
if not os.path.exists(des_dir):
    os.makedirs(des_dir)
""" 
read an image to get the shape
"""    
# shape_img=cv2.imread("data/allsolar_png512/20120601_0000_full_0.png").shape
shape_img=cv2.imread("data/allsolar_full_png512/20120101_0000_full_0.png").shape
shape=(4096,4096)
allFileNames = os.listdir(boxfiles_dir)
allFileNames=[ filename for filename in allFileNames if filename.endswith( '.csv' ) ]
for boxfile in allFileNames:
    boxdf=pd.read_csv(os.path.join(boxfiles_dir,boxfile),header=None)
    rows=boxdf.iloc[:,-4:].to_numpy()
    image = np.zeros((shape))
    for xmin, ymin, xmax, ymax in rows:
        try: 
            xmin, ymin, xmax, ymax=round(xmin),4096-round(ymin),round(xmax),4096-round(ymax)
        except:
            print("error "+ boxfile)
            continue #go to next for loop
        num_channels = 1 if len(image.shape) == 2 else image.shape[2]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(256,) * num_channels, thickness=-10)
    image=cv2.resize(image, shape_img[0:2])
    cv2.imwrite(os.path.join(des_dir,boxfile.split("box")[0]+"mask.png"), image)
