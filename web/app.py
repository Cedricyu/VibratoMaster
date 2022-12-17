from flask import Flask, Response, request
from flask import render_template, redirect
from pythonosc.udp_client import SimpleUDPClient
from record import record
from Metronome import play_sound
import WavePattern 
import linReg
import HarmRatioClf


# def send_osc_msg():

#     ip = "127.0.0.1"
#     port = 1337
#     client = SimpleUDPClient(ip, port)  # Create client
#     client.send_message("/something/address", 123)   # Send float message

# def send_osc_msg_a():

#     ip = "127.0.0.1"
#     port = 3819
#     client = SimpleUDPClient(ip, port)  # Create client
#     client.send_message("/toggle_roll", 0)   # Send float message





scoreWav = 80
string = "加油"
scoreLin = 70
socreHarm = 70

app = Flask(__name__)

@app.route('/<FUNCTION>')
def command(FUNCTION=None):
    exec(FUNCTION.replace("<br>", "\n"))
    return ""



@app.route('/menu')
def menu():

    return render_template('menu.html')

@app.route('/longbow')
def longbow():

    return render_template('longbow.html')

@app.route('/vibrato')


def vibrato():
    
    return render_template('vibrato.html')

@app.route('/basic01')
def basic01():

    return render_template('basic01.html')

@app.route('/basic02')
def basic02():

    return render_template('basic02.html')


@app.route('/basic03')
def basic03():

    return render_template('basic03.html')

@app.route('/prac')
def prac():

    return render_template('prac.html')


@app.route('/feedback')
def feedback():
    global scoreWav
    global scoreLin
    global scoreHarm

    scoreWav = WavePattern.test()
    scoreLin = linReg.linR()
    scoreHarm = HarmRatioClf.hrc()
    # scoreWav = 0
    # scoreLin = 0
    # scoreHarm = 0
	
    if( scoreWav + scoreHarm + scoreLin < 7 ):
        return render_template('good.html')

    return render_template('feedback.html', score = scoreWav, scoreLin = scoreLin, scoreHarm = scoreHarm)


def changeIndex():
    global score
    global string
    score = 90
    string = "表現良好"
    print(score)
    print(string)
    print("idex has changed")

@app.route('/')
def index():
    return render_template('index.html') 

@app.route("/wav")
def streamwav():
    def generate():
        with open("static/audio/session.wav", "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/x-wav")



if __name__ == '__main__':
	app.debug = True
	app.run('0.0.0.0')
