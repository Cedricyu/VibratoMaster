#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def dtw(s, t, window):
    n, m = len(s), len(t)
    w = np.max([window, abs(n-m)])
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
            
    result1 = dtw_matrix[i,j]
    print(result1)
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = abs(s[n-i] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    
    result2 = dtw_matrix[i,j]
    print(result2)
    return min(result1,result2)


# In[3]:


def similarity(s,t):
    A = 0.0
    B = 0.0
    C = 0.0
    Pxy = 0.0
    #print(s)
    #print(t)
    for i in range(len(s)):
        A+=s[i]*t[i]
        B+=s[i]*s[i]
        C+=t[i]*t[i]
   
    Pxy = A/((B*C)**0.5)
    if(math.isnan(Pxy)):
        return 0
    
    return Pxy


# In[4]:


import pandas as pd
df = pd.DataFrame(columns =  ["distance", "turn_point", "relation" ,"label"])


# In[5]:


import random
ran = random.randint(0, 100)
print(ran)

y = []
#sr = []
f0 = []
mu = []
sigma = []

lib_col = []
lib_row = []
lib_heigh = []

files = librosa.util.find_files("抖音txt/Wrist",
                                ext=['txt']) 
files = np.asarray(files)  

for n in files:
    with open(n) as f:
        polyShape = []
        for line in f:
            line = line.split() # to deal with blank 
            if line:            # lines (ie skip them)
                line = [float(i) for i in line]
                polyShape.append(line)
    f = np.array(polyShape)
    newarray = []




    #-------------------------------------------
    for i in range(0,len(f[0])):
        if(f[0][i]>400 and f[0][i]<900):
            newarray.append(f[0][i])
        #print(f[0][i])



    f0_note = [] # to note (A4-A5)
    for i in range(len(newarray)):
        f0_note.append(librosa.hz_to_note(newarray[i]))

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
    print("true_pitch",true_pitch)
    newarray = [i for i in newarray if (i > true_pitch -15 and i < true_pitch + 15)]
    #f0_tmp = np.array(f0_tmp)
    #-------------------------------------------




    #plt.plot(newarray)
    #plt.xlim(0,100)
    count=0
    T = 20
    ct = 0
    maxct = 0
    start = 0
    for i in range(1,len(newarray)-1):
        if(newarray[i] > newarray[i+1] and newarray[i] > newarray[i-1]):
            count+=1
            ct+=1
        elif(newarray[i] < newarray[i+1] and newarray[i] < newarray[i-1]):
            count+=1
            ct+=1
        if(i%T==0):
            if(ct > maxct):
                maxct = ct
                start = i-T
            ct =0

    x=np.linspace(0,2*np.pi,T)
    #predict = 3*np.sin(x) + true_pitch
    tmp = []
   
    window = []
    window_size = 5
    # plt.plot(newarray)
    # plt.show()
    
    if( T+start > len(newarray)):
        start -=T
    if( start < 0 ):
        start +=T
    print(start,len(newarray))
    for i in range(1,T):
        if(newarray[i+start] > newarray[i+start-1]):
            window.append(1)
        else:
            window.append(-1)
    slide_value = 0
    
    for i in range(T-window_size):
        slide =0 
        for j in range(window_size):
            slide+= window[i+j]
        window[i] = slide
    for i in range(1,T-window_size):
        #print("window ",i," =",window[i], "window ",i,"-1 =", window[i-1])
        if(window[i] < 0 and window[i-1] > 0):
            slide_value = i + int(window_size/2)
    start += slide_value
    if(start + slide_value+T > len(newarray)):
        start -= T
    # print("slide_value", slide_value)
    # print("start = ",start)
    # print("window =",window)
    # print("len of array =",len(newarray))
    for i in range(T):
        tmp.append(newarray[i+start])

    base = sum(tmp)/len(tmp)
    amplitude = (np.max(tmp)-np.min(tmp))/2
    predict = amplitude*np.sin(x + T/4) + base
    # plt.plot(tmp)
    # plt.plot(predict)
    distance = dtw(tmp,predict,window=2)
    relation = similarity(predict - base,tmp - base)
    # print("dtw = ",distance)
    # print("turn_points = ",count/len(newarray))
    # print("relation = ",relation)
    lib_row.append(distance)
    lib_col.append(count/len(newarray))
    lib_heigh.append(relation)
    plt.show()
    df = df.append({'distance':distance , 'turn_point':count/len(newarray), 'relation' : relation ,'label': 'Wrist' }, ignore_index=True)
    
# plt.scatter(lib_col,lib_heigh)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xlabel('turn_arround')
# plt.ylabel('relation')

print("dataframe = ",df)
# In[6]:


import random
# ran = random.randint(0, 100)
# print(ran)

y = []
#sr = []
f0 = []
mu = []
sigma = []

lib_col = []
lib_row = []
lib_heigh = []

files = librosa.util.find_files("Downloads/抖音txt/Good",
                                ext=['txt']) 
files = np.asarray(files)  

for n in files:
    with open(n) as f:
        polyShape = []
        for line in f:
            line = line.split() # to deal with blank 
            if line:            # lines (ie skip them)
                line = [float(i) for i in line]
                polyShape.append(line)
    f = np.array(polyShape)
    newarray = []




    #-------------------------------------------
    for i in range(0,len(f[0])):
        if(f[0][i]>400 and f[0][i]<900):
            newarray.append(f[0][i])
        #print(f[0][i])



    f0_note = [] # to note (A4-A5)
    for i in range(len(newarray)):
        f0_note.append(librosa.hz_to_note(newarray[i]))

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

    newarray = [i for i in newarray if (i > true_pitch - 15 and i < true_pitch + 15)]
    #f0_tmp = np.array(f0_tmp)
    #-------------------------------------------




    #plt.plot(newarray)
    #plt.xlim(0,100)
    T = 20
    count =0
    ct = 0
    maxct = 0
    start = 0
    for i in range(1,len(newarray)-1):
        if(newarray[i] > newarray[i+1] and newarray[i] > newarray[i-1]):
            count+=1
            ct+=1
        elif(newarray[i] < newarray[i+1] and newarray[i] < newarray[i-1]):
            count+=1
            ct+=1
        if(i%T==0):
            if(ct > maxct):
                maxct = ct
                start = i-T
            ct =0
#     T = 20
    #start = random.randint(0, len(newarray)-T)
    print("start = ",start)
    x=np.linspace(0,2*np.pi,T)
    #predict = 3*np.sin(x) + true_pitch
    tmp = []
    
    window = []
    window_size = 5
    
    for i in range(T):
        if(newarray[i+start] > newarray[i+start-1]):
            window.append(1)
        else:
            window.append(-1)
    slide_value = 0
    
    for i in range(T-window_size):
        slide =0 
        for j in range(window_size):
            slide+= window[i+j]
        window[i] = slide
    for i in range(1,len(window)):
        #print("window ",i," =",window[i], "window ",i,"-1 =", window[i-1])
        if(window[i] > 0 and window[i-1] < 0):
            slide_value = i + int(window_size/2)
    start += slide_value
    if(start + slide_value + T > len(newarray)):
        start -= T
    # print("slide_value", slide_value)
    # print("start = ",start)
    # print("window =",window)
    # print("len of array =",len(newarray))

   
    for i in range(T):
        tmp.append(newarray[i+start])
    base = sum(tmp)/len(tmp)
    amplitude = (np.max(tmp)-np.min(tmp))/2
    predict = amplitude*np.sin(x+T/4) + base
    # plt.plot(tmp)
    # plt.plot(predict)
    distance = dtw(tmp,predict,window=2)
    relation = similarity(predict - base,tmp - base)
    # print("dtw = ",distance)
    # print(count/len(newarray))
    # print("relation = ",relation)
    lib_row.append(distance)
    lib_col.append(count/len(newarray))
    lib_heigh.append(relation)
    df = df.append({'distance':distance , 'turn_point':count/len(newarray), 'relation' : relation ,'label': 'Good' }, ignore_index=True)
    #plt.show()
# plt.scatter(lib_col,lib_heigh)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xlabel('turn_arround')
# plt.ylabel('relation')
# plt.scatter(lib_col,lib_row)
# plt.show()


# In[7]:


# from sklearn.datasets import load_iris
# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state=0)


# # In[8]:


# X = df[['distance','turn_point','relation']]
# print(X)
# Y = df[['label']]
# print(Y)


# # In[9]:


# clf = DecisionTreeClassifier()
# clf = clf.fit(X, Y)


# In[10]:


# from sklearn import tree
# tree.plot_tree(clf)


# # In[11]:


# plt.figure()
# tree.plot_tree(clf,filled=True)  


# In[12]:


# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier()
# print(Y)
# neigh.fit(X, Y.values.ravel())

# print(neigh)


# In[13]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[14]:


# predictions = neigh.predict(X)
# print(accuracy_score(Y, predictions))


# In[15]:


test = pd.DataFrame(columns =  ["distance", "turn_point", "relation" ,"label"])


# In[16]:


# y = []
# sr = []
# t = []
# N = []
# xf = []
# yf = []
# mu = []

# files = librosa.util.find_files("Downloads/Wrist_test-20221203T082738Z-001/Wrist_test",
#                                 ext=['wav']) 
# files = np.asarray(files)  

# #print(files)

# for n in files:
    
#     y_tmp, sr_tmp = librosa.load(n)
#     f0_tmp, voiced_flag_tmp, voiced_probs_tmp = librosa.pyin(y_tmp, 
#     fmin=librosa.note_to_hz('G4'), fmax=librosa.note_to_hz('B5'))


#     y.append(y_tmp)
#     sr.append(sr_tmp)
    
#     t_tmp = librosa.get_duration(y=y_tmp,sr=sr_tmp)
#     t_tmp = int(t_tmp)
#     t.append(t_tmp)

#     N_tmp = sr_tmp * t_tmp
#     N.append(N_tmp)

#     y_tmp = np.real(rfft(y_tmp))
#     yf.append(np.abs(y_tmp))
    
#     xf.append(rfftfreq(N_tmp,1/sr_tmp))


#     # --------------------------------------
#     f0_tmp = [i for i in f0_tmp if (i > 0)]
#     f0_tmp = np.array(f0_tmp)
#     f0_tmp = f0_tmp[~np.isnan(f0_tmp)]
    
#     f0_note = [] # to note (A4-A5)
#     for i in range(len(f0_tmp)):
#         f0_note.append(librosa.hz_to_note(f0_tmp[i]))

#     note_cnt = np.array([0,0,0,0,0,0,0,0]) # A4 B4 C5 D5 E5 F5 G#5 A5
#     for i in range(len(f0_note)):
#         if(f0_note[i] == 'A4'):
#             note_cnt[0]+=1
#         elif(f0_note[i] == 'B4'):
#             note_cnt[1]+=1
#         elif(f0_note[i] == 'C5'):
#             note_cnt[2]+=1
#         elif(f0_note[i] == 'D5'):
#             note_cnt[3]+=1
#         elif(f0_note[i] == 'E5'):
#             note_cnt[4]+=1
#         elif(f0_note[i] == 'F5'):
#             note_cnt[5]+=1
#         elif(f0_note[i] == 'G♯5'):
#             note_cnt[6]+=1
#         elif(f0_note[i] == 'A5'):
#             note_cnt[7]+=1
    
#     true_note = np.argmax(note_cnt)
#     if(true_note == 0):
#         true_pitch = librosa.note_to_hz('A4')
#     elif(true_note == 1):
#         true_pitch = librosa.note_to_hz('B4')
#     elif(true_note == 2):
#         true_pitch = librosa.note_to_hz('C5')
#     elif(true_note == 3):
#         true_pitch = librosa.note_to_hz('D5')
#     elif(true_note == 4):
#         true_pitch = librosa.note_to_hz('E5')
#     elif(true_note == 5):
#         true_pitch = librosa.note_to_hz('F5')
#     elif(true_note == 6):
#         true_pitch = librosa.note_to_hz('G♯5')
#     elif(true_note == 7):
#         true_pitch = librosa.note_to_hz('A5')

#     f0_tmp = [i for i in f0_tmp if (i > true_pitch - 20 and i < true_pitch + 20)]
#     f0_tmp = np.array(f0_tmp)
#     #print(f0_tmp)
#     count =0 
#     start =0
#     ct = 0
#     maxct =0
#     T = 10
#     for i in range(len(f0_tmp)):
#         if(f0_tmp[i] > f0_tmp[i-1] and f0_tmp[i] >= f0_tmp[i-1] ):
#             ct+=1
#             count+=1
#         elif(f0_tmp[i] < f0_tmp[i-1] and f0_tmp[i] <= f0_tmp[i-1]):
#             count+=1
#             ct+=1
#         if(i%T==0):
#             if(ct > maxct):
#                 maxct = ct
#                 start = i-T
#             ct=0
#     x=np.linspace(0,2*np.pi,T)
#     #predict = 3*np.sin(x) + true_pitch
#     tmp = []
   
#     window = []
#     window_size = 5
#     newarray = f0_tmp
#     #plt.plot(newarray)
#     #plt.show()
    
#     if( T+start > len(newarray)):
#         start -=T
#     if( start < 0 ):
#         start +=T
#     print(start,len(newarray))
#     for i in range(1,T):
#         if(newarray[i+start] > newarray[i+start-1]):
#             window.append(1)
#         else:
#             window.append(-1)
#     slide_value = 0
    
#     for i in range(T-window_size):
#         slide =0 
#         for j in range(window_size):
#             slide+= window[i+j]
#         window[i] = slide
#     for i in range(1,T-window_size):
#         #print("window ",i," =",window[i], "window ",i,"-1 =", window[i-1])
#         if(window[i] < 0 and window[i-1] > 0):
#             slide_value = i 
#     start += slide_value
#     if(start + slide_value+T > len(newarray)):
#         start -= T
#     # print("slide_value", slide_value)
#     # print("start = ",start)
#     # print("window =",window)
#     # print("len of array =",len(newarray))
#     for i in range(T):
#         tmp.append(newarray[i+start])

#     base = sum(tmp)/len(tmp)
#     amplitude = (np.max(tmp)-np.min(tmp))/2
#     predict = amplitude*np.sin(x + T/4) + base
#     #plt.plot(tmp)
#     #plt.plot(predict)
#     distance = dtw(tmp,predict,window=2)
#     relation = similarity(predict - base,tmp - base)
#     # print("dtw = ",distance)
#     # print("turn_points = ",count/len(newarray))
#     # print("relation = ",relation)
#     lib_row.append(distance)
#     lib_col.append(count/len(newarray))
#     lib_heigh.append(relation)
#     #plt.show()
#     test = test.append({'distance':distance , 'turn_point':(count/len(newarray)), 'relation' : relation ,'label': 'Wrist' }, ignore_index=True)
    
#     #plt.plot(f0_tmp)
#     #plt.show()


# # In[17]:
# print("test= ",test)

# X_test = test[['distance','turn_point','relation']]
# print("Xtest = ",X_test)
# Y_test = test[['label']]
# print(Y_test)

X = df[['distance','turn_point','relation']]
print(X)
Y = df[['label']]
print(Y)


# # In[18]:


# # predictions = clf.predict(X_test)
# # print(predictions)
# print(accuracy_score(Y_test, predictions))


# In[19]:
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()

neigh.fit(X, Y.values.ravel())

# predictions = neigh.predict(X_test)
# print(predictions)
# print(accuracy_score(Y_test, predictions))


def test():
    y = []
    sr = []
    t = []
    N = []
    xf = []
    yf = []
    mu = []


    # files = np.asarray(files)  

    #print(files)

    
        
    y_tmp, sr_tmp = librosa.load("Good_test/A4_Good_2.wav")
    f0_tmp, voiced_flag_tmp, voiced_probs_tmp = librosa.pyin(y_tmp, 
    fmin=librosa.note_to_hz('G4'), fmax=librosa.note_to_hz('B5'))


    y.append(y_tmp)
    sr.append(sr_tmp)
    
    t_tmp = librosa.get_duration(y=y_tmp,sr=sr_tmp)
    t_tmp = int(t_tmp)
    t.append(t_tmp)

    N_tmp = sr_tmp * t_tmp
    N.append(N_tmp)

    y_tmp = np.real(rfft(y_tmp))
    yf.append(np.abs(y_tmp))
    
    xf.append(rfftfreq(N_tmp,1/sr_tmp))


    # --------------------------------------
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
    #print(f0_tmp)
    count =0 
    start =0
    ct = 0
    maxct =0
    T = 10
    for i in range(len(f0_tmp)):
        if(f0_tmp[i] > f0_tmp[i-1] and f0_tmp[i] > f0_tmp[i-1] ):
            ct+=1
            count+=1
        elif(f0_tmp[i] < f0_tmp[i-1] and f0_tmp[i] < f0_tmp[i-1]):
            count+=1
            ct+=1
        if(i%T==0):
            if(ct > maxct):
                maxct = ct
                start = i-T
            ct=0
    x=np.linspace(0,2*np.pi,T)
    #predict = 3*np.sin(x) + true_pitch
    tmp = []

    window = []
    window_size = 5
    newarray = f0_tmp
    # plt.plot(newarray)
    # plt.show()
    
    if( T+start > len(newarray)):
        start -=T
    if( start < 0 ):
        start +=T
    print(start,len(newarray))
    for i in range(1,T):
        if(newarray[i+start] > newarray[i+start-1]):
            window.append(1)
        else:
            window.append(-1)
    slide_value = 0
    
    for i in range(T-window_size):
        slide =0 
        for j in range(window_size):
            slide+= window[i+j]
        window[i] = slide
    for i in range(1,T-window_size):
        #print("window ",i," =",window[i], "window ",i,"-1 =", window[i-1])
        if(window[i] < 0 and window[i-1] > 0):
            slide_value = i + int(window_size/2)
    start += slide_value
    if(start + slide_value+T > len(newarray)):
        start -= T
    print("slide_value", slide_value)
    print("start = ",start)
    print("window =",window)
    print("len of array =",len(newarray))
    for i in range(T):
        tmp.append(newarray[i+start])

    base = sum(tmp)/len(tmp)
    amplitude = (np.max(tmp)-np.min(tmp))/2
    predict = amplitude*np.sin(x + T/4) + base
    # plt.plot(tmp)
    # plt.plot(predict)
    distance = dtw(tmp,predict,window=2)
    relation = similarity(predict - base,tmp - base)
    print("dtw = ",distance)
    
    print("turn_points = ",count/len(newarray))
    print("relation = ",relation)
    test = pd.DataFrame(columns =  ["distance", "turn_point", "relation" ,"label"])
    test = test.append({'distance':distance , 'turn_point':(count/len(newarray)), 'relation' : relation ,'label': 'Wrist' }, ignore_index=True)
    X_test = test[['distance','turn_point','relation']]
    print("Xtest = ",X_test)
    Y_test = test[['label']]
    print(Y_test)
    print(neigh.predict(X_test))
    # lib_row.append(distance)
    # lib_col.append(count/len(newarray))
    # lib_heigh.append(relation)
    # plt.show()
    return random.randint(0, 100)

# In[ ]:




