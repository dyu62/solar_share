import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import pandas as pd
import sys
import pickle
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
from datetime import date, timedelta,datetime
import random


    
    
class DatasetFolderWithPaths(VisionDataset):
    def __getitem__(self, index: int) -> Tuple[Any, Any,Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target,path) where target is class_index of the target class.
        """
        paths, target = self.samples[index]
        samples,att_masks=[],[]
        missflag_samples,missflag_masks=torch.zeros(self.time_length),torch.zeros(self.time_length)
        for i in range(self.time_length):
            if paths[i]=="miss.png":
                sample=torch.zeros_like(self.loader(os.path.join(self.root,paths[-1])))
                att_mask=torch.zeros(sample.shape[-2:])
                missflag_samples[i]=1
                missflag_masks[i]=1
            else:
                sample = self.loader(os.path.join(self.root,paths[i]))
                try: 
                    att_mask=(self.loader(os.path.join(self.root+"_boximage",paths[i][:13]+"_mask.png")))[0]
                except:
                    att_mask=torch.zeros(sample.shape[-2:])
                    missflag_masks[i]=1
            if self.transform is not None:
                sample = self.transform(sample)
            samples.append(sample)
            att_masks.append(att_mask)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return samples, target, att_masks, missflag_samples, missflag_masks, paths
    def __init__(
        self,
        root: str,
        pathnames:List,
        all_names:List,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        time_length=4,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        # pathnames=[os.path.join(root,i) for i in pathnames]
        # all_names=[os.path.join(root,i) for i in all_names]
        samples = self.make_dataset(pathnames,all_names, extensions, is_valid_file,time_length=time_length)

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]   
        self.time_length=time_length 

    @staticmethod
    def make_dataset(
        pathnames, #name of, like val only
        all_names, #all train, val, test names
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        time_length=4
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
        """
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]
        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        sortedpaths=(all_names)
        for i in range(len(sortedpaths)):
            if i < time_length-1:
                continue
            path=sortedpaths[i]
            if path in pathnames and is_valid_file(path):
                class_index=int(path.split(".")[-2][-1])
                paths=[sortedpaths[i-(time_length-1)+a] for a in range(time_length)]

                item = paths, class_index
                instances.append(item)
        return instances  
    def __len__(self) -> int:
        return len(self.samples)  
 
def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))
def cvsLoader(path: str):
    df = pd.read_csv(path, header = None)
    datanp=df.to_numpy()
    # datanp=cv2.resize(datanp, (224, 224))
    datanp=datanp[None,:]
    return torch.from_numpy(datanp)
def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return transforms.ToTensor()(img.convert("L")) #greyscale
    
def get_solar_dataset(data_dir,time_length,Seed,test_ratio=0.2):
    file_extention='.png'
    classes = {'pos':[], 'neg':[]}
    src = data_dir

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed)    
    allFileNames = os.listdir(src)
    allFileNames=[ filename for filename in allFileNames if filename.endswith( file_extention ) ]
    for file in allFileNames:
        if file.split(".")[-2].endswith(str(0)):
            classes["neg"].append(file)
        elif file.split(".")[-2].endswith(str(1)):
            classes["pos"].append(file)
    train_names,val_names,test_names=[],[],[]
    for key,paths in classes.items():
        np.random.shuffle(paths)
        val_test_split = int(np.around( test_ratio * len(paths) ))
        train_val_split = int(len(paths)-2*val_test_split)
        train_FileNames = paths[:train_val_split]
        val_FileNames = paths[train_val_split:train_val_split+val_test_split]
        test_FileNames = paths[train_val_split+val_test_split:]            

        print(f'{key} Total: ' +str(len(paths)))
        print('Train: ' +str(len(train_FileNames)))
        print('Val  : ' +str(len(val_FileNames)))
        print('Test : ' +str(len(test_FileNames)))                    
        train_names+=train_FileNames
        val_names+=val_FileNames
        test_names+=test_FileNames
    def deltadate(pathA,pathB):
        # '20120602_0000_full_0.png'
        dateA=(os.path.basename(pathA)).split('_')[0]+(os.path.basename(pathA)).split('_')[1]
        dateA=datetime.strptime(dateA, '%Y%m%d%H%M')
        dateB=(os.path.basename(pathB)).split('_')[0]+(os.path.basename(pathB)).split('_')[1]
        dateB=datetime.strptime(dateB, '%Y%m%d%H%M')
        deltaAB=dateB-dateA
        return deltaAB      
    def is_withinT(time_length,paths):
        flag=True
        delta0=deltadate(paths[0],paths[1])
        for i in range(time_length-2):
            deltaAB=deltadate(paths[1+i],paths[2+i])
            if deltaAB-delta0>timedelta(hours=1) and deltaAB-delta0<timedelta(hours=-1):
                flag=False
        return flag
    all_names=sorted(train_names+val_names+test_names)


    if data_dir=="data/allsolar_full_png512":
        std_deltaT=timedelta(hours=6) 
    elif data_dir=="data/allsolar_png512":
        std_deltaT=timedelta(hours=24) 
    i=0 
    l=len(all_names)
    while(i< l-1):
        current_deltaT=deltadate(all_names[i],all_names[i+1])
        if current_deltaT-std_deltaT>timedelta(hours=1) or current_deltaT-std_deltaT<timedelta(hours=-1):
            all_names.insert(i+1,"miss.png")
            i=i+1
        i+=1
    train_ds=DatasetFolderWithPaths(
        src,train_names,all_names, loader=pil_loader, extensions=['.png'],time_length=time_length
        ,transform=transforms.Compose([transforms.Resize((512, 512),antialias=False)])
        )   
    val_ds= DatasetFolderWithPaths(
        src,val_names,all_names, loader=pil_loader, extensions=['.png'],time_length=time_length
        ,transform=transforms.Compose([transforms.Resize((512, 512),antialias=False)])
        )  
    test_ds= DatasetFolderWithPaths(
        src,test_names,all_names, loader=pil_loader, extensions=['.png'],time_length=time_length
        ,transform=transforms.Compose([transforms.Resize((512, 512),antialias=False)])
        ) 
    return train_ds,val_ds,test_ds


if __name__ == '__main__':
    train_ds,val_ds,test_ds=get_solar_dataset("data/allsolar_full_png512",4,0,test_ratio=0.2)
    train_ds.__getitem__(0)
    train_ds[0]
    dl=torch.utils.data.DataLoader(train_ds,batch_size=16, shuffle=False,pin_memory=True) 
    a=next(iter(dl))
    a