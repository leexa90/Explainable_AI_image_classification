# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import gzip
from inception_v3_archisen import InceptionV3  #this inputes are uint8 to check for shalpley problems
import keras

from keras.layers import Input
Input_model = Input((299,299,3))  #299 can be anynumber or even None if undefined
Incept_model = InceptionV3(include_top=False, #chop of top
                weights='imagenet', #get pretrained weights
                input_tensor=Input_model, #trying uint8 image format
                input_shape=None, #ignore
                pooling=None, #ignore
                classes=1000) # classes: optional number only if top is true

output = Incept_model.output
output # shape of tensor is  (batch_size, Length,Width, Channels) 
output2 = keras.layers.GlobalAveragePooling2D()(output) 
output2 #global max pooling finds the max point in each channel. Frequently used, max finds the strongest signal. Average is also reasonable
output2 = keras.layers.Dropout(0.5)(output2)
output3 = keras.layers.Dense(96,activation='relu') (output2) # dense neural network with 50 layers
output4 = keras.layers.Dense(4, activation="softmax") (output3) # output 3 classes in probabiltites with softmax

from keras.models import Model
from keras.layers import Lambda
model = Model(Input_model,output4) #using logits layer
model.compile(loss = "categorical_crossentropy",
                    optimizer = 'adam',#ptimizers.SGD(lr=0.1),
                    metrics=['categorical_crossentropy',"accuracy"])

from keras.callbacks import EarlyStopping, ModelCheckpoint
model_path = 'transfer_model.h5'
callbacks = [
        EarlyStopping(
            monitor='val_categorical_crossentropy', 
            patience=3, # stop training at 3
            verbose=0),
        
        ModelCheckpoint(
            model_path , 
            monitor='val_categorical_crossentropy', 
            save_best_only=True, 
            verbose=0)
    ]
batch_size =32
from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=batch_size,
                         write_images=True)

skip = 2 #increase this to reduce sample size if low memory
c0 = np.load(gzip.GzipFile('./train/c_30.npy.gz'))[::skip] # tomato late blight
c1 = np.load(gzip.GzipFile('./train/c_31.npy.gz'))[::skip] # Tomato leaf mold 
c2 = np.load(gzip.GzipFile('./train/c_32.npy.gz'))[::skip] # Septrio leaf spot
c3 = np.load(gzip.GzipFile('./train/c_37.npy.gz'))[::skip] # healthy
y0 = np.array([[1,0,0,0],]*c0.shape[0]) #one hot encoded labels
y1 = np.array([[0,1,0,0],]*c1.shape[0])
y2 = np.array([[0,0,1,0],]*c2.shape[0])
y3 = np.array([[0,0,0,1],]*c3.shape[0])

# concateneate all classes together
X ,y = np.concatenate([c0,c1,c2,c3],axis=0), np.concatenate([y0,y1,y2,y3],axis=0)
del c0,c1,c2,c3
del y0,y1,y2,y3 #clear RAM

X_tr ,y_tr = X[0::2,], y[0::2,]
X_val , y_val = X[1::2,],y[1::2,]
del X ,y


import gc
gc.collect()

model.load_weights('transfer_model.h5') #trained with astar super computer for 30mins

# own shalpey
from skimage.segmentation import slic
dictt = {}
print (len(X_val))
for i in range(len(X_val))[::60]:
    temp = [[],[],[],[]]
    feature_names={'0':'late blight','1':'leaf mold','2':'leaf spot','3':'healthy'}
    img = X_val[i]
    import copy
    img_orig = copy.deepcopy(img)
    segments_slic = (slic(img, n_segments=56, compactness=10, sigma=0.5))
    preds_1  =model.predict(X_val[i:i+1])
    max_val = np.max(segments_slic)+1
    vec = np.array(range(max_val))
    Continue=False
    num =10000
    while Continue is False:
        random_array = np.random.randint(0,2,size=(num,max_val))
        if max(np.sum(random_array,1)) != max_val and min(np.sum(random_array,1)) !=0:
            Continue = True
    sample_index = map(lambda x : np.reshape(x,len(x)),map(lambda x : np.argwhere(x == 1),random_array))
    X_simple = np.array(map(lambda x : 1*np.isin(segments_slic,x),sample_index))
    print X_simple.shape
    gc.collect()
    import scipy
    M=max_val
    def shapley_kernel(x): #weights for each point
      return (M-1)/(scipy.misc.comb(M,len(x))*len(x)*(M-len(x)))
    W_mat = np.diag(map(shapley_kernel,sample_index))
    X= np.stack([X_simple,]*3,-1)*np.stack([img,]*num)+np.mean(img)*(np.stack([X_simple,]*3,-1)==0)
    gc.collect()
    Y_pred = model.predict(X,batch_size = 512/22)
    print Y_pred[0]
    Y_pred = np.log(Y_pred/(1-Y_pred)) #linearize probs
    print Y_pred[0]
    #φ = (X T W X) −1 X T W y
    for ii in range(len(Y_pred[0])):
        XtWX_inv = np.linalg.inv(np.matmul(np.matmul(random_array.T,W_mat),random_array))
        temp[ii] = np.matmul(np.matmul(np.matmul(XtWX_inv,random_array.T),W_mat),Y_pred[:,ii])
    dictt[i] = [img,temp, segments_slic ]
np.save('dictt60.npy',dictt)


