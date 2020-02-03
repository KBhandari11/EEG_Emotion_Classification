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
    data = np.zeros([40,19,8064])
    valence =[]
    arousal =[]
    for x in range (1,33):
        filename =  str(x) if x > 9 else (str(0)+ str(x))
        path = './data/s'+filename+'.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        valence = np.append(valence, valence_i)
        arousal = np.append(arousal, arousal_i)
        fig, ax = plt.subplots()
        ax.plot(range(1,20), data[0,:,:] )
        ax.set(xlabel='Channel (Ch)', ylabel='Hertz (Hz)',
            title=('Channel vs Hz'+ filename))
        ax.grid()
        fig.savefig("./pic/"+filename+".png")
        plt.show()
    return data, valence, arousal

data, valence, arousal = data_collection()

print(data)
np.savetxt('valence.txt', valence)
np.savetxt('arousal.txt', arousal)
print(data.shape, valence.shape, arousal.shape)

