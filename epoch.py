import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing
from ml_model import *

def data_filter(path, channel_rm):
    with open(path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p =u.load()
        labels = (p['labels'])
        data = (p['data'])
        valence, arousal = (labels[:,0], labels[:,1])
        data= np.delete(data, slice(32,47), axis=1)
        data_ch= np.delete(data, channel_rm, axis=1) 
    return data_ch, valence, arousal
def graph(epoch, loop):
     for x in range (0, loop):
        fig, ax = plt.subplots()
        ax.plot( epoch[x,:] )
        ax.set(xlabel='Time (ms)', ylabel='Hertz (Hz)',
            title=('Channel vs Hz'+ str(x)))
        ax.grid()
        plt.show()
def epoch(data):
    raw = np.zeros((40,19,63,128)) # [video, channel, time(s), time points] (40, 19, 63, 128)
    norm = np.zeros(raw.shape)
    min_max_scaler = preprocessing.MinMaxScaler()
    for x in range(0,40):
            for y in range(0,19):  
                raw[x,y,:,:]= np.array(np.split(data[x,y,:],raw.shape[2]))  # [:, :, time(s), time points]
                norm[x,y,:,:] = min_max_scaler.fit_transform(raw[x, y,:,:])
    raw = np.delete(raw, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    norm = np.delete(norm, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    return raw, norm  #(40, 19, 60, 128)

def feature_extraction(raw):
    psd = np.zeros((40, 19, 60, 64))
    f = np.zeros((40, 19, 60, 64))
    for x in range(0,40):
        for y in range(0,19):
            for z in range(0,60):
                f[x,y,z,:],psd[x,y,z,:]= signal.welch(raw[x,y,z,:],128,window="hamming",nperseg=127, average="mean")  # FFT to create psd of [19,40,60,64]
    return psd

def labeling(label):
    for i in range(label.shape[0]):
            label[i,0]= 0.9999 if (int(label[i,0]) > 5) else 0
            label[i,1]= 0.9999 if (int(label[i,1]) > 5) else 0
    return(label)    #(1280,) each; labeling according to the time(/s). 

def data_extract(raw, label):
    indices = np.arange(raw.shape[0])
    np.random.shuffle(indices)
    trainset = raw[indices]
    label = label[indices]
    res = int(''.join(map(str,indices.shape)))
    print(res)
    res = int(0.3 * res)
    trainset = trainset[0:res]
    label = label[0:res]
    return trainset,  label

def data_collection():
    epoch_data = np.zeros((40, 19, 60, 128), dtype='f')
    epoch_data_i = np.zeros((19, 60, 128), dtype='f')
    epoch_norm = np.zeros(epoch_data.shape, dtype='f')
    epoch_norm_i = np.zeros(epoch_data_i.shape, dtype='f')
    valence = np.zeros((40), dtype='f')
    arousal = np.zeros((40), dtype='f')
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30] 
    for x in range (1,33):
            filename =  str(x) if x > 9 else (str(0)+ str(x))
            path = '../data/s'+filename+'.dat'
            data, valence_i, arousal_i = data_filter(path, channel_rm )
            epoch_data_i, epoch_norm_i= epoch(data) # = (40, 19, 60, 128)  ; ie, [video, channel, time , timepoints)]
            #epoch_data_i = feature_extraction(epoch_norm_i)
            epoch_data = np.append(epoch_data, epoch_norm_i, axis=0)
            #epoch_data = np.append(epoch_data, epoch_data_i, axis=0)
            valence = np.append(valence, valence_i, axis=0)
            arousal = np.append(arousal, arousal_i, axis=0)
            label = np.vstack((valence, arousal)).T
    epoch_data= np.delete(epoch_data, slice(0,40), axis=0) #(1320, 19, 60, 128) =====> (1280, 19, 60, 128)
    label= np.delete(label, slice(0,40), axis=0) 
    label = labeling(label)         #(1280, 2)
    epoch_data = np.transpose(epoch_data, (0,2,3,1))  #(1280, 60, 128, 19)
    return epoch_data, label


train_data, train_label = data_collection()
print(train_data.shape, train_label.shape)
test_data, test_label = data_extract(train_data, train_label)
print(test_data.shape, test_label.shape)

kern_shape = (5,5)
model = CNN_Model(train_data,kern_shape, train_label, test_data, test_label )
model.train()