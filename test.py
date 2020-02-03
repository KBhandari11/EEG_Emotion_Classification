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
    raw_freq = np.zeros((40,19,63, 128))
    for x in range(0,40):
            for y in range(0,19):  
                raw_freq[x,y,:,:]= np.array(np.split(data[x,y,:],63))
    raw = np.delete(raw_freq, slice(0,3), axis=2)  
    print(raw.shape)
    raw = raw.transpose(1,2,0,3)
    print(raw.shape)
    raw = raw.reshape(raw.shape[0],-1, order='F')
    print(raw.shape)
    #norm[x,y,:]=list(map(lambda x, raw : (x - np.mean(raw)) / np.std(raw), raw))
    #freq[x,y,:]=list(map(lambda x: fft(x), time_epoch

def data_collection():
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30]
    for x in range (1,2):
        filename =  str(x) if x > 9 else (str(0)+ str(x))
        path = './data/s'+filename+'.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        print(data.shape)
        raw = epoch(data)
        print(raw.shape)
        #np.savetxt('./raw/'+str(filename)+'.txt', epoch,delimiter=',')
    return epoch

epoch= data_collection()
