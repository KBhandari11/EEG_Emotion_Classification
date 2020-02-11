import numpy as np
import pickle
import numpy
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
        data= numpy.delete(data, slice(32,47), axis=1)
        data_ch= numpy.delete(data, channel_rm, axis=1) 
    return data_ch, valence, arousal

def data_collection():
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30]
    for x in range (0,1):
        path = '../data/s01.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        fig, ax = plt.subplots()
        ax.plot( data[0,x,:] )
        ax.set(xlabel='Time (ms)', ylabel='Hertz (Hz)',
            title=('Channel vs Hz'))
        ax.grid()
        plt.show()
    return data

data = data_collection()
print(data.shape)
