
# coding: utf-8

# In[10]:


import h5py
import numpy as np
import json
from pprint import pprint

filename = 'sub_activitynet_v1-3.c3d.hdf5'
f = h5py.File(filename, 'r')


# In[11]:


# Build phrase objects
# Structure: 
#    dict: Key = phrase. Val = [C3D features, phrase features, vid_id].

full_objects = {}

for k in range(1,2):
    
    filename = 'train_vec'+str(k)+'.json'
    phrase_data = json.load(open(filename))
    
    vid_ids = list(phrase_data.keys())

    for i in vid_ids:
        c3d_features = np.array(f[i]['c3d_features'])

        phrases = phrase_data[i]["sentences"]
        vid_duration = phrase_data[i]["duration"]
        timestamps = np.array(phrase_data[i]["timestamps"])
        phrase_vecs = phrase_data[i]["vectors"]

        num_phrases = len(phrases)
        num_c3d_vecs = c3d_features.shape[0]

        indices = np.rint(num_c3d_vecs*timestamps/vid_duration).astype(int)
        for j in range(0,num_phrases):
            vid_vecs = c3d_features[np.arange(indices[j,0],indices[j,1]),:]
            sentence_vecs = np.array(phrase_vecs[j])
            full_objects[phrases[j]] = [vid_vecs, sentence_vecs, i]


# In[12]:


keys = full_objects.keys()
subset = {}
counter = 0
for i in keys:
    subset[i] = full_objects[i]
    counter += 1
    if counter == 100:
        break


# In[13]:


import pickle
with open('subset.pkl', 'wb') as fp:
    pickle.dump(subset, fp)


# In[16]:


subset_test = pickle.load(open('subset.pkl','rb'))


# In[17]:


print(subset_test)

