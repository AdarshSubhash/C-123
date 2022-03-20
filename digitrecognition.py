import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if (not os.environ.get("PYTHONHTTPSVERIFY","")and getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context=ssl._create_unverified_context

X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=["0","1","2","3","4","5","6","7","8","9"]
nclasses=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
xtrainscaled=xtrain/255.0
xtestscaled=xtest/255.0 

classify=LogisticRegression(solver="saga",multi_class="multinomial").fit(xtrainscaled,ytrain)

yprediction=classify.predict(xtestscaled)
accuracy=accuracy_score(ytest,yprediction)
print(accuracy)

capture=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=capture.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        Upper_left=(int(width/2-56),int(height/2-56))
        Bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,Upper_left,Bottom_right,(0,255,0),2)
        roi=gray[Upper_left[1]:Bottom_right[1],Upper_left[0]:Bottom_right]
        im_pil=Image.fromarrray(roi)
        image_bw=im_pil.convert("L")
        image_bw_resize=image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resize_inverted=PIL.ImageOps.invert(image_bw_resize)
        pixel_filter=20
        minpixel=np.percentile(image_bw_resize_inverted,pixel_filter)
        image_bw_resize_inverted_scaled=np.clip(image_bw_resize_inverted-minpixel,0,255)
        maxpixel=np.max(image_bw_resize_inverted)
        image_bw_resize_inverted_scaled=np.asarray(image_bw_resize_inverted_scaled)/maxpixel
        testsample=np.array(image_bw_resize_inverted_scaled).reshape(1,784)
        test_prediction=classify.predict(testsample)
        print("predictedclasses",test_prediction)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1)& 0xFF==ord("q"):
            break
    except Exception as e:
        pass
capture.release()        
cv2.destroyAllWindows()