from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

import tensorflow as tf


seed=3


calech_dir='data'

image_w=64
image_h=64

pixels=image_h*image_w*3

X=[]
filenames=[]
files=glob.glob(calech_dir+"/*/*.*")

for i, f in enumerate(files):
    img=Image.open(f)
    img=img.convert("RGB")
    img=img.resize((image_w,image_h))
    data=np.asarray(img)
    filenames.append(f)
    X.append(data)


X=np.array(X)
X=X.astype(float)/255
model=load_model('./model/human_classify.model')

prediction=model.predict(X)

np.set_printoptions(formatter={'float':lambda  x:"{0:0.3f}".format(x)})

cnt=0

# for i in prediction:
#     if i>=0.5:print("해당"+filenames[cnt]+"여자입니다.")
#     else: print("해당"+filenames[cnt]+"남자입니다.")
#     cnt+=1