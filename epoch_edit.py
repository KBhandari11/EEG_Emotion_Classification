import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from ml_model import *


##-----------------------------------
def data_filter(path):
    with open(path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p =u.load()
        labels = (p['labels'])
        data = (p['data'])
        valence, arousal = (labels[:,0], labels[:,1])
        data= np.delete(data, slice(32,47), axis=1)
    return data, valence, arousal
def graph(epoch, loop):
     for x in range (0, loop):
        fig, ax = plt.subplots()
        ax.plot( epoch[x,:] )
        ax.set(xlabel='Time (ms)', ylabel='Hertz (Hz)',
            title=('Channel vs Hz'+ str(x)))
        ax.grid()
        plt.show()
def epoch(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    raw = np.zeros((40,32,63,128)) # [video, channel, time(s), time points] (40, 32, 63, 128)
    for x in range(0,40):
            for y in range(0,32):  
                raw[x,y,:,:]= np.array(np.split(data[x,y,:],raw.shape[2]))  # [:, :, time(s), time points]
                raw[x,y,:,:] = min_max_scaler.fit_transform(raw[x, y,:,:])  # data normalization using Min Max scalar
    raw = np.delete(raw, slice(0,3), axis=2)    # Removed 3s starting baseline (40, 32, 60, 128)
    return raw #(40, 32, 60, 128)
def labeling(label_1, x):
    m = label_1.shape[0] * x 
    label_a= np.ones((m,2))
    for i in range(label_1.shape[0]):
        for j in range(i*x, i*x+x):
            label_a[j,0]= 1 if (int(label_1[i][0]) > 5) else 0
            label_a[j,1]= 1 if (int(label_1[i][1]) > 5) else 0
    return(label_a)    #(1280,) each; labeling according to the time(/s). 
def data_collection():
    epoch_data = np.ones((40, 32, 60, 128), dtype='f') #while using raw/norm
    epoch_data_i = np.ones((32, 60, 128), dtype='f')
    valence = np.zeros((40), dtype='f')
    arousal = np.zeros((40), dtype='f')
    print("new")
    for x in range (1,33):
            filename =  str(x) if x > 9 else (str(0)+ str(x))
            path = '../data/s'+filename+'.dat'
            data, valence_i, arousal_i = data_filter(path)
            epoch_data_i= epoch(data) # = (40, 32, 60, 128)  ; ie, [video, channel, time , timepoints)]
            epoch_data = np.append(epoch_data, epoch_data_i, axis=0)  #normalized
            valence = np.append(valence, valence_i, axis=0)
            arousal = np.append(arousal, arousal_i, axis=0)
            label = np.vstack((valence, arousal)).T
    print("xxx")
    epoch_data= np.delete(epoch_data, slice(0,40), axis=0) #(1320, 32, 60, 128) =====> (1280, 32, 60, 128)
    label= np.delete(label, slice(0,40), axis=0) 
    label_feature = labeling(label, 60 * 128) 
    label = labeling(label, 60)         #(1280, 2)``
    epoch_data=np.transpose(epoch_data,(0,2,3,1))  #=====> (1280, 60, 128, 32)
    m, n, o, p = epoch_data.shape
    data_channel =  np.reshape(epoch_data,(m*n*o , p)) 
    channel_rm= feature_selection(data_channel, label_feature[:,0],num_channel=19)
    epoch_data= np.delete(epoch_data, channel_rm, axis=3)
    epoch_data = np.reshape(epoch_data,(m*n,o, p)) #=====> (1280 * 60, 128, 32) for Time domain
    epoch_data = epoch_data[:,:,:, np.newaxis] # (1280 * 60, 128, 32, 1)
    print(epoch_data.shape)
    return epoch_data, label
def feature_selection(data, label, num_channel):
    print("test")
    channel_rm_list = []
    channel_all_list = set(list(range(32)))
    sfs = SFS(LinearRegression(),
        k_features=num_channel,
        forward=True,
        floating=False,
        scoring = 'r2',
        cv = 0) 
    sfs.fit(data, label)
    x = sfs.k_feature_names_     # to get the final set of features
    channel_list = set([int(a) for a in list(x)])
    channel_rm_list = list(channel_all_list.difference(channel_list))
    return channel_rm_list
##-----------------------------------

##<driver program>
train_data, train_label = data_collection()
"""
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
"""