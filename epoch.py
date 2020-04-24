import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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
    norm = np.zeros(raw.shape)
    min_max_scaler = preprocessing.MinMaxScaler()
    for x in range(0,40):
            for y in range(0,19):  
                raw[x,y,:,:]= np.array(np.split(data[x,y,:],raw.shape[2]))  # [:, :, time(s), time points]
                norm[x,y,:,:] = min_max_scaler.fit_transform(raw[x, y,:,:])
    raw = np.delete(raw, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    norm = np.delete(norm, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    return raw, norm  #(40, 19, 60, 128)

def feature_extraction(raw):
    psd = np.zeros((40, 19, 60, 64))
    f = np.zeros((40, 19, 60, 64))
    min_max_scaler = preprocessing.MinMaxScaler()
    for x in range(0,40):
        for y in range(0,19):
            for z in range(0,60):
                f[x,y,z,:],psd[x,y,z,:]= signal.welch(raw[x,y,z,:],128,window="hamming",nperseg=127, average="median")  # FFT to create psd of [19,40,60,64]
            psd[x,y,:,:] = min_max_scaler.fit_transform(psd[x, y,:,:])
    return psd

def labeling(label):
    for i in range(label.shape[0]):
            label[i,0]= 1 if (int(label[i,0]) > 5) else 0
            label[i,1]= 1 if (int(label[i,1]) > 5) else 0
    return(label)    #(1280,) each; labeling according to the time(/s). 


def data_collection():
    #epoch_data = np.zeros((40, 19, 60, 64), dtype='f') #while using PS
    epoch_data = np.zeros((40, 19, 60, 128), dtype='f') #while using raw/norm
    epoch_data_i = np.zeros((19, 60, 128), dtype='f')
    epoch_norm = np.zeros(epoch_data.shape, dtype='f')
    epoch_norm_i = np.zeros(epoch_data_i.shape, dtype='f')
    valence = np.zeros((40), dtype='f')
    arousal = np.zeros((40), dtype='f')
    channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30] 
    for x in range (1,33):
            filename =  str(x) if x > 9 else (str(0)+ str(x))
            path = '../data/s'+filename+'.dat'
            data, valence_i, arousal_i = data_filter(path, channel_rm )
            epoch_data_i, epoch_norm_i= epoch(data) # = (40, 19, 60, 128)  ; ie, [video, channel, time , timepoints)]
            #epoch_data_i = feature_extraction(epoch_data_i) # = (40, 19, 60, 64) 
            epoch_data = np.append(epoch_data, epoch_norm_i, axis=0)  #normalized
            #epoch_data = np.append(epoch_data, epoch_data_i, axis=0)  #feature
            valence = np.append(valence, valence_i, axis=0)
            arousal = np.append(arousal, arousal_i, axis=0)
            label = np.vstack((valence, arousal)).T
    epoch_data= np.delete(epoch_data, slice(0,40), axis=0) #(1320, 19, 60, 128) =====> (1280, 19, 60, 128)
    label= np.delete(label, slice(0,40), axis=0) 
    label = labeling(label)         #(1280, 2)
    epoch_data = np.transpose(epoch_data, (0,2,3,1))  #(1280, 60, 128, 19)
    return epoch_data, label


train_data, train_label = data_collection()
train_data, test_data, train_label, test_label = train_test_split(train_data, train_label, test_size=0.25)
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
kern_shape = (5,5)
model = CNN_Model(train_data,kern_shape, train_label[:,0], test_data, test_label[:,0] )
history = model.train()
	
# list all data in history
print(history.history.keys())
print("Accuracy: ")
print("%.4f%% (+/- %.4f%%)" % (np.mean(history.history['accuracy']), np.std(history.history['accuracy'])))
print("AUC: ")
print("%.4f%% (+/- %.4f%%)" % (np.mean(history.history['auc']), np.std(history.history['auc'])))
print("Loss: ")
print("%.4f%% (+/- %.4f%%)" % (np.mean(history.history['loss']), np.std(history.history['loss'])))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for auc
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()