import librosa
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
import math
import glob
import os
from itertools import chain
import scipy.stats as stats
#import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# y, sr, f0, mu, sigma
y = []
sr = []
f0 = []
mu = []
sigma = []

files = librosa.util.find_files("Good",
                                ext=['wav']) 
files = np.asarray(files)  

for n in files:
    y_tmp, sr_tmp = librosa.load(n)
    y.append(y_tmp)
    sr.append(sr_tmp)

    f0_tmp, voiced_flag_tmp, voiced_probs_tmp = librosa.pyin(y_tmp, 
    fmin=librosa.note_to_hz('G4'), fmax=librosa.note_to_hz('B5'))

    f0_tmp = [i for i in f0_tmp if (i > 0)]
    f0_tmp = np.array(f0_tmp)
    f0_tmp = f0_tmp[~np.isnan(f0_tmp)]

    f0_note = [] # to note (A4-A5)
    for i in range(len(f0_tmp)):
        f0_note.append(librosa.hz_to_note(f0_tmp[i]))

    note_cnt = np.array([0,0,0,0,0,0,0,0]) # A4 B4 C5 D5 E5 F5 G#5 A5
    for i in range(len(f0_note)):
        if(f0_note[i] == 'A4'):
            note_cnt[0]+=1
        elif(f0_note[i] == 'B4'):
            note_cnt[1]+=1
        elif(f0_note[i] == 'C5'):
            note_cnt[2]+=1
        elif(f0_note[i] == 'D5'):
            note_cnt[3]+=1
        elif(f0_note[i] == 'E5'):
            note_cnt[4]+=1
        elif(f0_note[i] == 'F5'):
            note_cnt[5]+=1
        elif(f0_note[i] == 'G♯5'):
            note_cnt[6]+=1
        elif(f0_note[i] == 'A5'):
            note_cnt[7]+=1

    true_note = np.argmax(note_cnt)
    if(true_note == 0):
        true_pitch = librosa.note_to_hz('A4')
    elif(true_note == 1):
        true_pitch = librosa.note_to_hz('B4')
    elif(true_note == 2):
        true_pitch = librosa.note_to_hz('C5')
    elif(true_note == 3):
        true_pitch = librosa.note_to_hz('D5')
    elif(true_note == 4):
        true_pitch = librosa.note_to_hz('E5')
    elif(true_note == 5):
        true_pitch = librosa.note_to_hz('F5')
    elif(true_note == 6):
        true_pitch = librosa.note_to_hz('G♯5')
    elif(true_note == 7):
        true_pitch = librosa.note_to_hz('A5')

    f0_tmp = [i for i in f0_tmp if (i > true_pitch - 20 and i < true_pitch + 20)]
    f0_tmp = np.array(f0_tmp)

    f0.append(f0_tmp)
    mu.append(f0_tmp.mean())
    sigma.append(f0_tmp.std())

mu = np.array(mu)
sigma = np.array(sigma)
# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(mu[:,np.newaxis], sigma)






#----------------------------------------------------------------------
# user input
def linR():
    from octave import scale
    score = 0
    print("lenscale",len(scale))
    for i in range(len(scale)):
        arr = np.array(scale[i])
        # plt.plot(arr)
        # plt.show()
        if len(arr) == 0:
            print("Invalid input. Please try again.")
        else:
            res = arr.std() - regr.predict([[arr.mean()]])
            # print("std ", arr.std())
            # print("predict ", regr.predict([[arr.mean()]]))
            # print("res ",res)
            if abs(res) < 0.7:
                print("Good")
            else:
                print("Thumb")
                score+=1
    print(score)
    return score 
#----------------------------------------------------------------------

#linR()