import librosa

y, sr = librosa.load('a_minir_1.wav')
f0, voiced_flag, voiced_probs = librosa.pyin(y, 
fmin=librosa.note_to_hz('G4'),fmax=librosa.note_to_hz('B5'))

f0 = [i for i in f0 if ((i > librosa.note_to_hz('A4') - 20 and i < librosa.note_to_hz('A4') + 20) or
                        (i > librosa.note_to_hz('B4') - 20 and i < librosa.note_to_hz('B4') + 20) or
                        (i > librosa.note_to_hz('C5') - 20 and i < librosa.note_to_hz('C5') + 20) or
                        (i > librosa.note_to_hz('D5') - 20 and i < librosa.note_to_hz('D5') + 20) or
                        (i > librosa.note_to_hz('E5') - 20 and i < librosa.note_to_hz('E5') + 20) or
                        (i > librosa.note_to_hz('F5') - 20 and i < librosa.note_to_hz('F5') + 20) or
                        (i > librosa.note_to_hz('G♯5') - 20 and i < librosa.note_to_hz('G♯5') + 20) or
                        (i > librosa.note_to_hz('A5') - 20 and i < librosa.note_to_hz('A5') + 20))]



scale = []
scale.append([i for i in f0 if (i > librosa.note_to_hz('A4') - 20 and i < librosa.note_to_hz('A4') + 20)])
scale.append([i for i in f0 if (i > librosa.note_to_hz('B4') - 20 and i < librosa.note_to_hz('B4') + 20)])
scale.append([i for i in f0 if (i > librosa.note_to_hz('C5') - 20 and i < librosa.note_to_hz('C5') + 20)])
scale.append([i for i in f0 if (i > librosa.note_to_hz('D5') - 20 and i < librosa.note_to_hz('D5') + 20)])
scale.append([i for i in f0 if (i > librosa.note_to_hz('E5') - 20 and i < librosa.note_to_hz('E5') + 20)])
scale.append([i for i in f0 if (i > librosa.note_to_hz('F5') - 20 and i < librosa.note_to_hz('F5') + 20)])
scale.append([i for i in f0 if (i > librosa.note_to_hz('G♯5') - 20 and i < librosa.note_to_hz('G♯5') + 20)])
scale.append([i for i in f0 if (i > librosa.note_to_hz('A5') - 20 and i < librosa.note_to_hz('A5') + 20)])