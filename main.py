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
    for x in range(0,40):
            for y in range(0,19):  
                raw[x,y,:,:]= np.array(np.split(data[x,y,:],raw.shape[2]))  # [:, :, time(s), time points]
    raw = np.delete(raw, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    raw = raw.reshape(40,19,-1)             
    raw = raw.transpose(1,0,2)                     # (40, 19, 7680)
    raw = raw.reshape(raw.shape[0],-1, order='F')       # Reduce the dimension to 2x2 array (19, 307200)
    return raw
def feature_extraction(raw):
    data= np.zeros((19,2400, 128))
    psd = np.zeros((19,2400,64))
    f = np.zeros(psd.shape)
    for x in range(0,19):
        data[x,:]= np.array(np.split(raw[x,:],2400))                      # Split according to video trial [19,2400,128]
        for y in range(0,2400):
            f[x,y,:], psd[x,y,:]= signal.welch(data[x,y,:],128,nperseg=127)  # FFT to create frequency of [19,2400,64] and psd of [19,40,60,64]
    return data, psd
def normalize(raw):
    norm = np.zeros(raw.shape)
    for x in range(0,raw.shape[0]):
        norm[x,:]= preprocessing.scale(raw[x,:])
    return norm
def labeling(valence, arousal):
    m = valence.shape
    valence_all= np.ones((2400))
    arousal_all= np.ones((2400)) 
    for i in range(0,40):
        for j in range(i*60,i*60+60):
            valence_all[j]= 1 if (int(valence[i]) > 5) else 0
            arousal_all[j]= 1 if (int(arousal[i]) > 5) else 0
    return(valence_all,arousal_all )    #(2400,) each; labeling according to the time(/s). 
def input_dimension(param):
    value = []
    for x in param:
        x = x.transpose(1,0,2)
        x = np.expand_dims(x, axis=3)
        value.append(x)
    return value
def data_collection():
    raw = np.zeros((19,2400,128))
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30] 
    for x in range (5,6):
        filename =  str(x) if x > 9 else (str(0)+ str(x))
        path = '../data/s'+filename+'.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        valence, arousal = labeling(valence_i, arousal_i)  #(2400,) each; labeling according to the time(/s). 
        epoch_data = epoch(data) # = [19, 307200]  ; ie, [channel, epoch * timepoints)]
        #min_max_scaler = preprocessing.MinMaxScaler()
        #norm = min_max_scaler.fit_transform(raw)
        epoch_norm= normalize(epoch_data)
        raw ,psd_raw = feature_extraction(epoch_data)
        norm ,psd_norm = feature_extraction(epoch_norm)
        param = list = [raw,psd_raw, norm ,psd_norm]
        raw,psd_raw, norm ,psd_norm = input_dimension(param)
        print(norm)
    return raw ,psd_raw, norm ,psd_norm,valence, arousal  

raw,psd_raw, norm ,psd_norm, valence, arousal  = data_collection() #(2400, 19, 128, 1), (2400, 19, 64, 1), (2400, 19, 128, 1), (2400, 19, 64, 1) 
kern_shape = (5,5)
#model = CNN_Model(norm, kern_shape, valence, arousal)
#model.train()