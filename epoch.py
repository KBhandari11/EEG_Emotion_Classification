import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt


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
    print(raw.shape)
    raw = raw.reshape(40,19,-1)             
    print(raw.shape)
    raw = raw.transpose(1,0,2)                     # (40, 19, 7680)
    raw = raw.reshape(raw.shape[0],-1, order='F')       # Reduce the dimension to 2x2 array (19, 307200)
    print(raw.shape)    
    return raw

def data_collection():
    raw = np.zeros((19,307200))
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30]
    for x in range (1,2):
        filename =  str(x) if x > 9 else (str(0)+ str(x))
        path = '../data/s'+filename+'.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        raw = epoch(data) # = [19, 307200]  ; ie, [channel, epoch * timepoints)]
        np.savetxt('./'+str(filename)+'.txt', raw,delimiter=',')
    return epoch

raw= data_collection()
