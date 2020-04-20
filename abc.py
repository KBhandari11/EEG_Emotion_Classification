import numpy as np 


valence = np.arange(5)
arousal = np.arange(5,10)
label = np.vstack((valence, arousal)).T
print(label)