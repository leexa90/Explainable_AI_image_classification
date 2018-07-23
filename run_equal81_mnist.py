# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import numpy as np
import gzip
import keras

from keras.layers import Input



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
#%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau



#https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
model = Sequential()
Input 
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)



batch_size = 64



model.load_weights('mnist.h5') #trained with astar super computer for 30mins

X_train = mnist.train.images
X_train = np.reshape(X_train,(X_train.shape[0],28,28,1))
X_val = mnist.test.images
X_val = np.reshape(X_val,(X_val.shape[0],28,28,1))[::1]
y_train = mnist.train.labels
y_val = mnist.test.labels[::1]


pred= model.predict(X_val)


# In[ ]:

pred2 = np.argmax(pred,1)
print  ('Accuracy of test set',np.mean(pred2==np.argmax(y_val,1)))
print  ('Inaccurately predicted',np.sum(pred2!=np.argmax(y_val,1)))
print ('Total test set size',len(pred))


# own shalpey
#from skimage.segmentation import slic
import gc
filename= 'dictt_BetterSampling_mnist.npy'
def slic(img, n_segments=49, compactness=10, sigma=0.5):
    new_img = np.copy(img[:,:,0])
    counter =0
    for i in range(0,28,4)[:]:
        for j in range(0,28,4)[:]:
            a,b = 4,4
            new_img[i:i+a,j:j+b] = counter
            counter += 1
    return new_img.astype(np.int32)
import itertools
import scipy
dictt ={}# np.load('dictt_BetterSampling_100_81_preds_oneside.npy')#{}
import os
if filename+'.gz' in os.listdir('.'):
    dictt= np.load(gzip.GzipFile(filename+'.gz')).item()
print (len(X_val))
for i in range(len(X_val))[::]:
    if i in dictt.keys() and i !=0 : 
       print i,'in dictt.keys()'
       continue
    temp = [[],]*10
    feature_names={'0':'late blight','1':'leaf mold','2':'leaf spot','3':'healthy'}
    img = X_val[i]
    import copy
    img_orig = copy.deepcopy(img)
    segments_slic = (slic(img, n_segments=34, compactness=10, sigma=0.5))
    preds_1  =model.predict(X_val[i:i+1]) 
    if np.argmax( preds_1[0]) == np.argmax(y_val[i]) and i !=0:
       print np.argmax( preds_1[0]) , np.argmax(y_val[i])
       continue
    max_val = np.max(segments_slic)+1 #segment is from 0 to n-1
    vec = np.array(range(max_val))
    Continue=False
    num =10000
    num_comb = 0
    sample_index = []
    counter = 0
    while num_comb < num:
        num_comb += scipy.misc.comb(max_val,counter)
        counter += 1
    for num_of_chosen in range(1,counter):
        if num_of_chosen < counter-1:
            sample_index += list(itertools.combinations(range(max_val), max_val-num_of_chosen))
            #sample_index += list(itertools.combinations(range(max_val), num_of_chosen))
        else:
            sample_index += list(itertools.combinations(range(max_val), max_val-num_of_chosen))[::3] 
            #random_array = np.random.randint(0,2,size=(num-len(sample_index),max_val))
            #sample_index += map(lambda x : np.reshape(x,len(x)),
            #                   map(lambda x : np.argwhere(x == 1),random_array))
    X_simple = np.array(map(lambda x : 1*np.isin(segments_slic,x),sample_index)).astype(np.float32)
    random_array = []
    for l2 in sample_index:
        temp2 = np.zeros(max_val)
        for j2 in l2:
            temp2[j2] = 1
        random_array += [temp2,]
    random_array  = np.stack(random_array,0).astype(np.float32)
    print (X_simple.shape)
    gc.collect()
    import scipy
    M=max_val
    
    def shapley_kernel(x): #weights for each point
      return (M-1)/(scipy.misc.comb(M,len(x))*len(x)*(M-len(x)))
    W_mat = np.diag(map(shapley_kernel,sample_index)).astype(np.float32)
    X= np.stack([X_simple,]*1,-1)*np.stack([img,]*len(X_simple))+.0*np.mean(img)*(np.stack([X_simple,]*1,-1)==0).astype(np.float32)
    gc.collect()
    Y_pred = model.predict(X,batch_size = 512/4)  
#    Y_pred1 = model.predict(X[1::2],batch_size = 512/4)
#    Y_pred = np.concatenate([Y_pred0,Y_pred1],0)
    print Y_pred.shape
    print (Y_pred[0])
    Y_pred = np.log((0.000001+Y_pred)/(1.000001-Y_pred))#-np.log(1./9) #linearize probs
    print len(Y_pred)
    print (preds_1)
    #φ = (X T W X) −1 X T W y
    for ii in range(len(Y_pred[0])):
        XtWX_inv = np.linalg.inv(np.matmul(np.matmul(random_array.T,W_mat),random_array))
        temp[ii] = np.matmul(np.matmul(np.matmul(XtWX_inv,random_array.T),W_mat),Y_pred[:,ii])
    dictt[i] = [img,temp, segments_slic,y_val[i] ] 
    print( temp[ii])
    if len(dictt) %10 == 0:
      np.save(filename,dictt)
      print len(dictt)
np.save(filename,dictt)
for i  in range(len(X_val))[::]:
    if i in dictt.keys():
       print i,'in dictt.keys()'
       continue
    temp = [[],]*10
    feature_names={'0':'late blight','1':'leaf mold','2':'leaf spot','3':'healthy'}
    img = X_val[i]
    import copy
    img_orig = copy.deepcopy(img)
    segments_slic = (slic(img, n_segments=34, compactness=10, sigma=0.5))
    preds_1  =model.predict(X_val[i:i+1])
    if np.argmax( preds_1[0]) != np.argmax(y_val[i]):
       continue
    max_val = np.max(segments_slic)+1 #segment is from 0 to n-1
    vec = np.array(range(max_val))
    Continue=False
    num =10000
    #num_comb = 0
    #sample_index = []
    #counter = 0
    #while num_comb < num:
    #    num_comb += 1*scipy.misc.comb(max_val,counter)
    #    counter += 1
    #for num_of_chosen in range(1,counter):
    #    if num_of_chosen < counter-1:
    #        sample_index += list(itertools.combinations(range(max_val), max_val-num_of_chosen))
            #sample_index += list(itertools.combinations(range(max_val), num_of_chosen))
    #    else:
    #         sample_index += list(itertools.combinations(range(max_val), max_val-num_of_chosen))[::3]

            #random_array = np.random.randint(0,2,size=(num-len(sample_index),max_val))
            #sample_index += map(lambda x : np.reshape(x,len(x)),
            #                   map(lambda x : np.argwhere(x == 1),random_array))
    #X_simple = np.array(map(lambda x : 1*np.isin(segments_slic,x),sample_index)).astype(np.float32)
    #random_array = []
    #for l2 in sample_index:
    #    temp2 = np.zeros(max_val)
    #    for j2 in l2:
    #        temp2[j2] = 1
    #    random_array += [temp2,]
    #random_array  = np.stack(random_array,0).astype(np.float32)
    print (X_simple.shape)
    gc.collect()
    import scipy
    M=max_val
    
    def shapley_kernel(x): #weights for each point
      return (M-1)/(scipy.misc.comb(M,len(x))*len(x)*(M-len(x)))
    W_mat = np.diag(map(shapley_kernel,sample_index)).astype(np.float32)
    X= np.stack([X_simple,]*1,-1)*np.stack([img,]*len(X_simple))+.0*np.mean(img)*(np.stack([X_simple,]*1,-1)==0).astype(np.float32)
    gc.collect()
    Y_pred = model.predict(X,batch_size = 512/4) 
#    Y_pred1 = model.predict(X[1::2],batch_size = 512/4)
#    Y_pred = np.concatenate([Y_pred0,Y_pred1],0)
    print Y_pred.shape
    print (Y_pred[0])
    Y_pred = np.log((0.000001+Y_pred)/(1.000001-Y_pred))  #- np.log(1.0/9)#linearize probs
    print len(Y_pred)
    print (preds_1)
    #φ = (X T W X) −1 X T W y
    for ii in range(len(Y_pred[0])):
        XtWX_inv = np.linalg.inv(np.matmul(np.matmul(random_array.T,W_mat),random_array))
        temp[ii] = np.matmul(np.matmul(np.matmul(XtWX_inv,random_array.T),W_mat),Y_pred[:,ii])
    dictt[i] = [img,temp, segments_slic,y_val[i] ] 
    if len(dictt) %10 == 0:
      np.save(filename,dictt)
      print len(dictt)
np.save(filename,dictt)



