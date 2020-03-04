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
    return raw  #(40, 19, 60, 128)

def feature_extraction(raw):
    psd = np.zeros((40, 19, 60, 64))
    f = np.zeros(psd.shape)
    for x in range(0,40):
        for y in range(0,19):
            for z in range(0,60):
                f[x,y,:], psd[x,y,:]= signal.welch(raw[x,y,z,:],128,nperseg=127)  # FFT to create frequency of [19,2400,64] and psd of [19,40,60,64]
    return psd
    """
    #Graph 
    # Define delta lower and upper limits
    low = [0.5, 4, 8, 13, 30]
    high =[4, 8, 13, 30, 47]
    col =['skyblue','blue', 'green','red','cyan']
    f=f[1,1,1,:]
    psd=psd[1,1,1,:]
    for x in range(0,5):
        # Find intersecting values in frequency vector
        idx = np.logical_and(f >= low[x], f <= high[x])
        # Plot the power spectral density and fill the delta area
        plt.plot(f, psd,lw=2, color='k')
        plt.fill_between(f, psd, where=idx, color= col[x])
        # Plot the power spectrum
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd.max() * 1.1])
        plt.title("Welch's periodogram")
        plt.xlim([0, f.max()])
        plt.show()
    """

def normalize(raw):
    norm = np.zeros(raw.shape)
    for x in range(0,raw.shape[0]):
        norm[x,:]= preprocessing.scale(raw[x,:])
    return norm

def labeling(valence, arousal):
    m = valence.shape
    valence_all = np.zeros(valence.shape, dtype='f')
    arousal_all = np.zeros(arousal.shape, dtype='f')
    for i in range(0,40):
            valence_all[i]= 0.9999 if (int(valence[i]) > 5) else 0
            arousal_all[i]= 0.9999 if (int(arousal[i]) > 5) else 0
    return(valence_all,arousal_all )    #(40,) each; labeling according to the time(/s). 

def data_collection():
    raw = np.zeros((19,307200), dtype='f')
    epoch_data = np.zeros((40, 19, 60, 128), dtype='f')
    epoch_data_i = np.zeros((19, 60, 128), dtype='f')
    valence = np.zeros((40), dtype='f')
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30] 
    for x in range (1,33):
        filename =  str(x) if x > 9 else (str(0)+ str(x))
        path = '../data/s'+filename+'.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        #valence, arousal = labeling(valence_i, arousal_i)  #(2400,) each; labeling according to the time(/s). 
        epoch_data_i= epoch(data) # = (40, 19, 60, 128)  ; ie, [video, channel, time , timepoints)]
        #min_max_scaler = preprocessing.MinMaxScaler()
        #norm = min_max_scaler.fit_transform(raw)
        #epoch_norm= normalize(epoch_data)
        #psd_raw = feature_extraction(epoch_data)
        #norm ,psd_norm = feature_extraction(epoch_norm)
        #np.savetxt('./'+str(filename)+'.txt', raw,delimiter=',')
        epoch_data = np.append(epoch_data, epoch_data_i, axis=0)
        valence = np.append(valence, valence_i, axis=0)
    return epoch_data, valence, arousal_i


raw, valence, arousal= data_collection()
raw= np.delete(raw, slice(0,40), axis=0)
valence= np.delete(valence, slice(0,40), axis=0)


valence = np.fromiter((map(lambda x: x/10, valence)), dtype=float)
print(raw.shape)
#raw,psd_raw, valence, arousal  = data_collection()
kern_shape = (5,5)
model = CNN_Model(raw, kern_shape, valence, arousal)
model.train()