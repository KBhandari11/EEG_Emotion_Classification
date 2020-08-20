import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
from scipy.integrate import simps
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from ml_model import *

##-----------------------------------
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
def graph(f, data):
    print(data.shape)
    f= np.delete(f, slice(45,65), axis=0) 
    f= np.delete(f, slice(0,4), axis=0) 
    plt.figure()
    for x in range(0, 1280):
        plt.semilogy(f, data[x,0,0,:])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    plt.ylim([0, data[x,0,:].max() * 1.1]) 
    plt.title("Welch's periodogram")
    plt.legend()
    plt.xlim([0, f.max()])
    plt.show()
def epoch(data):
    raw = np.zeros((40,32,63,128)) # [video, channel, time(s), time points] (40, 19, 63, 128)
    norm = np.zeros(raw.shape)
    min_max_scaler = preprocessing.MinMaxScaler()
    for x in range(0,40):
            for y in range(0,32):  
                raw[x,y,:,:]= np.array(np.split(data[x,y,:],raw.shape[2]))  # [:, :, time(s), time points]
                norm[x,y,:,:] = min_max_scaler.fit_transform(raw[x, y,:,:])
    raw = np.delete(raw, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    norm = np.delete(norm, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 19, 60, 128)
    return raw, norm  #(40, 19, 60, 128)

def frequency_band(f, psd):
    low = [0.5, 4, 8, 14, 30]
    high = [4, 8, 14, 30, 45]
    m, n, o, p = psd.shape
    freq_param = np.zeros((m,n,o,5))
    freq_res = f[1] - f[0]  
    min_max_scaler = preprocessing.MinMaxScaler()

    for x in range(m):
        for y in range(n):
            psd[x,y,:,:] = min_max_scaler.fit_transform(psd[x, y,:,:])  # min max feature scaling

            """ for z in range(o):
                for w in range(5):
                    idx_delta = np.logical_and(f >= low[w], f <= high[w])
                    #freq_param[x,y,z,w] = simps(psd[x,y,z,idx_delta], dx=freq_res)
                    freq_param[x,y,z,w] = entropy(psd[x,y,z,idx_delta], base=2)
    return freq_param """
    return psd
def labeling(label_1):
    m = label_1.shape[0] * 60
    label= np.ones((m,2))
    for i in range(label_1.shape[0]):
        for j in range(i*60, i*60+60):
            label[j,0]= 1 if (int(label_1[i][0]) > 5) else 0
            label[j,1]= 1 if (int(label_1[i][1]) > 5) else 0
    return(label)    #(1280,) each; labeling according to the time(/s). 
def data_collection():
    epoch_data = np.ones((40, 32, 60, 65), dtype='f') #while using PS
    epoch_data_i = np.zeros((32, 60, 128), dtype='f')
    epoch_norm = np.zeros(epoch_data.shape, dtype='f')
    epoch_norm_i = np.zeros(epoch_data_i.shape, dtype='f')
    valence = np.zeros((40), dtype='f')
    arousal = np.zeros((40), dtype='f')
    #channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30] #19 channels
    channel_rm = [] # 32 channels all
    #channel_rm = [1,4,5,8,9,12,14,17,21,22,26,27,30]
    for x in range (1,33):
            filename =  str(x) if x > 9 else (str(0)+ str(x))
            path = '../data/s'+filename+'.dat'
            data, valence_i, arousal_i = data_filter(path, channel_rm )
            epoch_data_i, epoch_norm_i= epoch(data) # = (40, 19, 60, 128)  ; ie, [video, channel, time , timepoints)]
            sf = 128
            win = sf
            f, epoch_data_i = signal.welch(epoch_data_i,sf,window="hamming",nperseg=win, axis=-1, nfft=None)
            epoch_data_i = frequency_band(f, epoch_data_i) # = (40, 19, 60, 64) 
            epoch_data = np.append(epoch_data, epoch_data_i, axis=0)  #feature
            valence = np.append(valence, valence_i, axis=0)
            arousal = np.append(arousal, arousal_i, axis=0)
            label = np.vstack((valence, arousal)).T
    epoch_data= np.delete(epoch_data, slice(0,40), axis=0) #(1320, 19, 60, 128) =====> (1280, 19, 60, 128)
    epoch_data= np.delete(epoch_data, slice(45,65), axis=3) #(1320, 19, 60, 128) =====> (1280, 19, 60, 45)
    epoch_data= np.delete(epoch_data, slice(0,4), axis=3) #(1320, 19, 60, 128) =====> (1280, 19, 60, 41)
    print(epoch_data.shape)
    #graph(f,epoch_data)
    label= np.delete(label, slice(0,40), axis=0) 
    label = labeling(label)         #(1280, 2)
    print(label.shape)
    epoch_data=np.transpose(epoch_data,(0,2,3,1))  #=====> (1280, 60, 128, 19)
    m, n = epoch_data.shape[0], epoch_data.shape[1]
    epoch_data = np.reshape(epoch_data,(m*n,41, 32)) #=====> (1280 * 60, 128, 19) for Frequency domain
    epoch_data = epoch_data[:,:,:, np.newaxis] # (1280 * 60, 128, 19, 1)
    print(epoch_data)
    return epoch_data, label


##-----------------------------------

##<driver program>
train_data, train_label = data_collection()
train_data, test_data, train_label, test_label = train_test_split(train_data, train_label, test_size=0.05)
train_data, crosstest_data, train_label, crosstest_label = train_test_split(train_data, train_label, test_size=0.25)
print(train_data.shape, train_label.shape, crosstest_data.shape, crosstest_label.shape, test_data.shape, test_label.shape)
kern_shape = (5,5)
model = CNN_Model(train_data,kern_shape, train_label[:,0], crosstest_data, crosstest_label[:,0], test_data, test_label[:,0] )
history, test_new = model.train()
print(test_new)

##<--------Output-------->
	
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
