import librosa
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
import math
import glob
import os
from itertools import chain
import scipy.stats as stats
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#----------------------------------------------------------------------
# user input
def linR():
    # y, sr, f0, mu, sigma
    y = []
    sr = []
    f0 = []
    mu = []
    sigma = []

    files = librosa.util.find_files("C:/Users/David/Desktop/cmat/VibratoMaster/web/Good",
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







    #-------------------------------------------------------------------------------------------------------
    y_test, sr_test = librosa.load('C:/Users/David/Desktop/cmat/VibratoMaster/web/Good_test/A4_Good_2.wav')
    f0_test, voiced_flag_test, voiced_probs_test = librosa.pyin(y_test, 
    fmin=librosa.note_to_hz('G4'), fmax=librosa.note_to_hz('B5'))

    f0_test = [i for i in f0_test if (i > 0)]
    f0_test = np.array(f0_test)
    f0_test = f0_test[~np.isnan(f0_test)]

    f0_note_test = [] # to note (A4-A5)
    for i in range(len(f0_test)):
        f0_note_test.append(librosa.hz_to_note(f0_test[i]))

    note_cnt_test = np.array([0,0,0,0,0,0,0,0]) # A4 B4 C5 D5 E5 F5 G#5 A5
    for i in range(len(f0_note_test)):
        if(f0_note_test[i] == 'A4'):
            note_cnt_test[0]+=1
        elif(f0_note_test[i] == 'B4'):
            note_cnt_test[1]+=1
        elif(f0_note_test[i] == 'C5'):
            note_cnt_test[2]+=1
        elif(f0_note_test[i] == 'D5'):
            note_cnt_test[3]+=1
        elif(f0_note_test[i] == 'E5'):
            note_cnt_test[4]+=1
        elif(f0_note_test[i] == 'F5'):
            note_cnt_test[5]+=1
        elif(f0_note_test[i] == 'G♯5'):
            note_cnt_test[6]+=1
        elif(f0_note_test[i] == 'A5'):
            note_cnt_test[7]+=1

    true_note_test = np.argmax(note_cnt_test)
    if(true_note_test == 0):
        true_pitch_test = librosa.note_to_hz('A4')
    elif(true_note_test == 1):
        true_pitch_test = librosa.note_to_hz('B4')
    elif(true_note_test == 2):
        true_pitch_test = librosa.note_to_hz('C5')
    elif(true_note_test == 3):
        true_pitch_test = librosa.note_to_hz('D5')
    elif(true_note_test == 4):
        true_pitch_test = librosa.note_to_hz('E5')
    elif(true_note_test == 5):
        true_pitch_test = librosa.note_to_hz('F5')
    elif(true_note_test == 6):
        true_pitch_test = librosa.note_to_hz('G♯5')
    elif(true_note_test == 7):
        true_pitch_test = librosa.note_to_hz('A5')

    f0_test = [i for i in f0_test if (i > true_pitch_test - 20 and i < true_pitch_test + 20)]
    f0_test = np.array(f0_test)
    
    
    """
    pos = 0
    chunk = []
    for i in range (8):
        chunk_tmp = []
        if i != 7:
            for j in range(int(len(f0_test) / 8)):
                chunk_tmp.append(f0_test[j + pos])
            pos += int(len(f0_test) / 8)
            chunk.append(chunk_tmp)
        else:
            for j in range(int(len(f0_test) / 8 + len(f0_test) % 8)):
                chunk_tmp.append(f0_test[j + pos]) 
            chunk.append(chunk_tmp)


    for i in range(len(chunk)):
        chunk[i] = np.array(chunk[i])

    mu_test = []
    sigma_test = []
    for i in range(len(chunk)):
        mu_test.append(chunk[i].mean())
        sigma_test.append(chunk[i].std())

    mu_test = np.array(mu_test)
    regr = linear_model.LinearRegression()
    regr.fit(mu_test.reshape(-1,1), sigma_test)
    """
    if len(f0_test) == 0:
        print("Invalid input. Please try again.")
    else:
        res = f0_test.std() - regr.predict([[f0_test.mean()]])
        if abs(res) < 0.5:
            print("Good")
        else:
            print("Thumb")
#----------------------------------------------------------------------


linR()


"""
plt.scatter(mu, sigma, facecolor="none", edgecolor='blue', s=15)
plt.plot(mu, regr.intercept_ + regr.coef_ * mu, linewidth=3, color='red')
plt.xlabel('f0 mean')
plt.ylabel('sigma')
plt.title('Linear Regression (Good')
plt.show()
"""


"""
#---------------------------------------------------------------
# Test




files = librosa.util.find_files("C:/Users/David/Desktop/cmat/VibratoMaster/web/Good_test",
                              ext=['wav']) 
files = np.asarray(files)  

for n in files:
    y_tmp, sr_tmp = librosa.load(n)


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


    plt.scatter(f0_tmp.mean(), f0_tmp.std(), color='green', s=15)

    res = f0_tmp.std() - regr.predict([[f0_tmp.mean()]])

    if abs(res) < 0.5:
        print("Good")
    else:
        print("Thumb")

plt.show()
"""

