import cv2
import matplotlib.pyplot as plt
def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(299,299))
    return img

import os
import numpy as np

def save_numpy(str_):
    data1 = [x for x in os.listdir('./c_'+str_)]
    all = []
    for i in data1[:]:
        print i
        img = read_img('./c_'+str_+'/'+i)
        #plt.imsave('i0'+i,img)
        if all == []:
            all += [[img],]
        else:
            all += [[img],]
    all = np.concatenate(all)
    print 'done'
    np.save('c_'+str_,all.astype(np.uint8))

for i in (30,31,32,37):
    i = str(i)
    save_numpy(i)
die

data1 = [x for x in os.listdir('./c_5')]
all = []
for i in data1:
    img = read_img('./c_5/'+i)
    #plt.imsave('i1'+i,img)
    if all == []:
        all = [img]
    else:
        all = np.append(all,[img],axis=0)
np.save('c_1',all.astype(np.float32))

data1 = [x for x in os.listdir('./c_6')]
all = []
for i in data1:
    img = read_img('./c_6/'+i)
    #plt.imsave('i1'+i,img)
    if all == []:
        all = [img]
    else:
        all = np.append(all,[img],axis=0)
np.save('c_3',all.astype(np.float32))
