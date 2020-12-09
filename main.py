import csv
import os

from PIL import Image,ImageFilter
import imgaug
import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa
import shutil
base_directory=r'D:\PycharmProjects\pythonProject'

def move():
    print('copy')
    i=0
    train_dir=os.path.join(base_directory,r'jiangnan2020\train\train')
    with open(os.path.join(base_directory,r'jiangnan2020\train.csv'),'r') as file:
        csv_reader=csv.reader(file)
        csv_list=list(csv_reader)
        csv_list.pop(0)
        for row in csv_list:
            base_name=row[0]+'.jpg'
            path=os.path.join(train_dir,base_name)
            if int(row[1])==0:shutil.move(path,r'D:\PycharmProjects\pythonProject\jiangnan2020\ten\0')
            if int(row[1])==1:shutil.move(path,r'D:\PycharmProjects\pythonProject\jiangnan2020\ten\1')

def copy():
    train_dir = os.path.join(base_directory, r'jiangnan2020\train\train')
    with open(os.path.join(base_directory, r'\jiangnan2020\ten\0'), 'r') as file:
        csv_reader = csv.reader(file)
        csv_list = list(csv_reader)
        csv_list.pop(0)
        i=0
        for row in csv_list:
            i=i+1
            if i>1000:break
            base_name = row[0] + '.jpg'
            path = os.path.join(train_dir, base_name)
            shutil.copy(path,'D:\PycharmProjects\pythonProject\jiangnan2020\ten1\0')

if __name__=='__main__':
    copy()