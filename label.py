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
    return data_ch, np.array(valence), np.array(arousal)
def graph(epoch, loop):
     for x in range (0, loop):
        fig, ax = plt.subplots()
        ax.plot( epoch[x,:] )
        ax.set(xlabel='Time (ms)', ylabel='Hertz (Hz)',
            title=('Channel vs Hz'+ str(x)))
        ax.grid()
        plt.show()
def epoch(data):
    raw = np.zeros((40,19,63,128)) # [video, channel, time(s), time points]
    for x in range(0,40):
            for y in range(0,19):  
                raw[x,y,:,:]= np.array(np.split(data[x,y,:],63))  # [:, :, time(s), time points]
    raw = np.delete(raw, slice(0,3), axis=3)    # Removed 3s starting baseline
    raw = raw.reshape(40,19,-1)
    raw = raw.transpose(1,0,2)
    raw = raw.reshape(raw.shape[0],-1, order='F')       # Reduce the dimension to 2x2 array
    return raw
def labeling(valence, arousal):
    m = valence.shape
    valence_all= np.ones((2400))
    arousal_all= np.ones((2400)) 
    for i in range(0,40):
        for j in range(i*60,i*60+60):
            valence_all[j]= 1 if (int(valence[i]) > 5) else 0
            arousal_all[j]= 1 if (int(arousal[i]) > 5) else 0
    return(valence_all,arousal_all )    #(2400,) each; labeling according to the time(/s). 


def data_collection():
    raw = np.zeros((19,307200))
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30]
    for x in range (1,3):
        filename =  str(x) if x > 9 else (str(0)+ str(x))
        path = '../data/s'+filename+'.dat'
        data, valence, arousal = data_filter(path, channel_rm )
        raw = epoch(data) # = [19, 307200]  ; ie, [channel, epoch * timepoints)]
        valence, arousal  = labeling(valence,arousal)
        print(valence.shape)
        print(arousal.shape)
        #np.savetxt('./aro'+str(filename)+'.txt', arousal,delimiter=',')
        #np.savetxt('./val'+str(filename)+'.txt', valence,delimiter=',')
    return epoch

raw= data_collection()

