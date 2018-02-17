import numpy as np
import torch
import torch.nn as nn
import torch.util.data as data
from torch.autograd import Variable
import pickle
import random

class Dataset(data.Dataset):

    def __init__(self, filename, anchor_is_phrase = True):
        pairs_dict = pickle.load( open( filename, "rb" ) )
        self.triplet_dict = make_triplets(pairs_dict, anchor_is_phrase)


    def make_triplets(self, pairs_dict, anchor_is_phrase, num_negative = 3):
        triplet_dict = {}
        counter = 0
        for key in pairs_dict:
            video, caption, vid_id = pairs_dict[key]
            for j in range(num_negative):
                rand_key = random.choice(list(pairs_dict.keys))
                while rand_key == key:
                    rand_key = random.choice(list(pairs_dict.keys))
                rvideo, rcaption, rvid_id = pairs_dict[rand_key]
                triple = None
                if anchor_is_phrase:
                    triple = (caption, video, rvideo)
                else:
                    triple = (video, caption, rcaption)
                triplet_dict[counter] = triple
                counter += 1

    def __len__(self):
        return len(self.triplet_dict)

    def __getitem__(self, index):
        return self.triplet_dict[index]






