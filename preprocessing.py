import tensorflow as tf
from PIL import Image

import os,glob,sys,numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

img_dir='data'
categories=['female','male']
np_classes=len(categories)
image_w=64
image_h=64

pixel=image_h*image_w*3

x=[]
y=[]

for idx,cat in enumerate(categories):
    img_dir_detail=img_dir+"\\"+cat
    files=glob.glob(img_dir_detail+"\\*")
    # print(files)
    # print("-------------")
    # print(img_dir_detail)

    for i,f in enumerate(files):
        try:
            img=Image.open(f)
            img=img.convert("RGB")
            img=img.resize((image_w,image_h))
            data=np.asarray(img)


            x.append(data)
            y.append(idx)
            if i% 300==0:
                print(cat,":",f)

        except:
            print(cat,str(i)+"번째에서 에러")

X=np.array(x)
Y=np.array(y)

X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
xy=(X_train,X_test,Y_train,Y_test)
np.save("_image_data.npy",xy)


