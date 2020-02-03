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
def graph(epoch, loop):
     for x in range (0, loop):
        fig, ax = plt.subplots()
        ax.plot( epoch[x,:] )
        ax.set(xlabel='Time (ms)', ylabel='Hertz (Hz)',
            title=('Channel vs Hz'+ str(x)))
        ax.grid()
        plt.show()

def data_collection():
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30]
    for i in range (1,2):
        epoch = np.zeros((40,19,63))
        filename =  str(i) if i > 9 else (str(0)+ str(i))
        path = './data/s'+filename+'.dat'
        data, valence_i, arousal_i = data_filter(path, channel_rm )
        for x in range(0,40):
            for y in range(0,19):  
                epoch[x,y,:]= np.split(data[x,y,:],63)
        epoch = numpy.delete(epoch, slice(0,3), axis=2)   
        epoch = epoch.transpose(1,2,0)
        epoch=epoch.reshape(epoch.shape[0],-1, order='F') 
        print(epoch.shape)
        np.savetxt('./epoch/'+str(filename)+'.txt', epoch,delimiter=',')
    return epoch

epoch= data_collection()
