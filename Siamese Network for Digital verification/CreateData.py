import cv2
from tqdm import tqdm
import os
import random
from random import shuffle
import numpy as np


TRAIN_DIR = './data/training/024'


GENUINE_DIR  = os.path.join(TRAIN_DIR,'genuine_24')
FORGED_DIR   = os.path.join(TRAIN_DIR,'forged_24')



X_left = []
X_right = []
label = []


def create_train_genuine():

    genuine = []

    for img in tqdm(os.listdir(GENUINE_DIR)):

        

        path = os.path.join(GENUINE_DIR,img)
        
        img  = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,(100,100))
        genuine.append(np.array(img))
        return genuine


def create_train_forge():

    forged = []

    for img in tqdm(os.listdir(FORGED_DIR)):

        path = os.path.join(FORGED_DIR,img)
        img  = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,(100,100))

        forged.append(np.array(img))
        return forged


def create_train_similar(genuine):

    for _ in range(20):

        x = random.choice(genuine)
        y = random.choice(genuine)

        X_left.append(x)
        X_right.append(y)
        label.append(np.array([1]))


def create_train_dissimilar(genuine,forged):

    for _ in range(20):

        x = random.choice(genuine)
        y = random.choice(forged)

        X_left.append(x)
        X_right.append(y)
        label.append(np.array([0]))        

        

        


gen   = create_train_genuine()
forge   = create_train_forge()


create_train_dissimilar(gen,forge)
create_train_similar(gen)

np.save('X_left.npy',np.array(X_left))
np.save('X_right.npy',np.array(X_right))
np.save('label.npy',np.array(label))

        
