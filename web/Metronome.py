import simpleaudio, time 
strong_beat = simpleaudio.WaveObject.from_wave_file('metronome.wav')

play = False
def play_sound():
    
    strong_beat.play()
    time.sleep(0.5)