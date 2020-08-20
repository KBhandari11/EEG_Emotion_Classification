import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from sklearn import preprocessing
import random

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
    min_max_scaler = preprocessing.MinMaxScaler()
    raw = np.zeros((40,19,63,128)) # [video, channel, time(s), time points] (40, 19, 63, 128)
    for x in range(0,40):
            for y in range(0,19):  
                raw[x,y,:,:]= np.array(np.split(data[x,y,:],63))  # [:, :, time(s), time points]
                #raw[x,y,:,:] = min_max_scaler.fit_transform(raw[x, y,:,:])
    raw = np.delete(raw, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    #raw = raw.reshape(40,19,-1)             
    #raw = raw.transpose(1,0,2)                     # (40, 19, 7680)
    #raw = raw.reshape(raw.shape[0],-1, order='F')       # Reduce the dimension to 2x2 array (19, 307200)
    return raw


def data_collection():
    raw = np.zeros((19,307200))
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30] 
    for x in range (1,2):
        filename =  str(x) if x > 9 else (str(0)+ str(x))
        path = '../data/s'+filename+'.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        raw = epoch(data) # = [19, 307200]  ; ie, [channel, epoch * timepoints)]
        #one liner psd calculation
        print(raw.shape)
        sf = 128
        win = sf
        f, psd = signal.welch(raw,sf,window="hamming",nperseg=win, axis=-1, nfft=None)
        #print(psd)
        print(psd.shape)
        min_max_scaler = preprocessing.MinMaxScaler()
        psd = np.reshape(psd, (45600,65))
        psd = psd.T
        psd = min_max_scaler.fit_transform(psd)
        #Graph 
        # Define lower and upper limits of all the powerband
        plt.figure()
        for z in range(100):
            #x = random.randint(0,45600)
            print(psd[:,z])
            plt.semilogy(f, psd[:,z])
        #for x in range(45000,45600):
          #  plt.semilogy(f, psd[:,x])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd.max() * 1.1]) 
        plt.title("Welch's periodogram")
        plt.legend()
        plt.xlim([0, f.max()])
        print(psd.max())
        print(f.max())
        plt.show()
        #freq = psd(raw)
        #print(f)
        #print(Pxx_den)
        #np.savetxt('./'+str(filename)+'.txt', raw,delimiter=',')
        
    return epoch

raw= data_collection()
