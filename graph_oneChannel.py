import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing

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
                raw[x,y,:,:]= np.array(np.split(data[x,y,:],63))  # [:, :, time(s), time points]
    raw = np.delete(raw, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    raw = raw.reshape(40,19,-1)             
    raw = raw.transpose(1,0,2)                     # (40, 19, 7680)
    raw = raw.reshape(raw.shape[0],-1, order='F')       # Reduce the dimension to 2x2 array (19, 307200)
    return raw

def psd(raw):
    freq = np.zeros((19,40, 60, 128))
    freq_1 = np.zeros((19,40, 7680))
    psd = np.zeros((19,40,60,64))
    f = np.zeros(psd.shape)
    min_max_scaler = preprocessing.MinMaxScaler()
    for x in range(0,19):
        freq_1[x,:]= np.array(np.split(raw[x,:],40)) 
    for x in range(0,19):
        for y in range(0,40):
            freq[x,y,:]= np.array(np.split(freq_1[x,y,:],60)) 
            for z in range(0,60):
                f[x,y,z,:], psd[x,y,z,:]= signal.welch(freq[x,y,z,:],128,nperseg=127)
    print(psd.shape)
    print(f.shape)
    f = np.reshape(f,(19,153600))
    psd = np.reshape(psd, (19,153600))

    #Graph 
    # Define lower and upper limits of all the powerband
    for x in range(19):
        plt.figure()
        plt.semilogy(f[x,:], psd[x,:])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd.max() * 1.1]) 
        plt.title("Welch's periodogram")
        plt.xlim([0, f.max()])
        print(psd.max())
        print(f.max())
        plt.show()
    return freq
def data_collection():
    raw = np.zeros((19,307200))
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30] 
    for x in range (1,2):
        filename =  str(x) if x > 9 else (str(0)+ str(x))
        path = '../data/s'+filename+'.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        raw = epoch(data) # = [19, 307200]  ; ie, [channel, epoch * timepoints)]
        freq = psd(raw)
        #print(f)
        #print(Pxx_den)
        #np.savetxt('./'+str(filename)+'.txt', raw,delimiter=',')
    return epoch

raw= data_collection()
