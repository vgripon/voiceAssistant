import pyaudio
import numpy as np
import array
import os
from gtts import gTTS
import time
import sys
import subprocess
import playsound

import webrtcvad

# Parameters for PyAudio
rate = 16000
chunk = int(rate * 30 / 1000)
frmat = pyaudio.paInt16
channels = 1
SAMPLE_RATE = rate

vad = webrtcvad.Vad(3)

speechFrames = []

from gtts import gTTS
def voice(text):
    print("Ordinateur:", text)
    tts = gTTS(text=text, lang = "fr")
    filename = "/tmp/output.mp3"
    tts.save(filename)
    playsound.playsound("/tmp/output.mp3")

import openai

init = [{"role": "system", "content": "You are a helpful assistant. Please provide short answers of one sentence maximum."}]
queries = []
lastUpdate = time.time()
def discuss(text):
    global queries, lastUpdate
    if time.time() - lastUpdate > 60:
        queries = []
    queries.append({"role":"user", "content": "S'il te plait réponds en une phrase maximum. " + text})
    
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=queries
    )

    answer = result["choices"][0]["message"]["content"]
    queries.append({"role": "assistant", "content": answer})
    lastUpdate = time.time()
    voice(answer)

import whisper
model = whisper.load_model(sys.argv[1] if len(sys.argv) > 1 else "small")
modelTiny = whisper.load_model("base.en")

def ding():
    playsound.playsound("ding.mp3")

valid = 0
def analyseSpeech(speech):
    global valid, timeOff
    if time.time() - valid < 20:
        transcribedText = model.transcribe(speech, language = "french")
    else:
        transcribedText = modelTiny.transcribe(speech, language = "english")
    if transcribedText["text"] not in ["", " Sous-titres réalisés par la communauté d'Amara.org"]:
        print(transcribedText["text"])
        done = False
        try:
            parts = transcribedText["text"].split()
            if parts[1][:6] == "minute":
                timeOff = time.time() + int(parts[0])*60 + int(parts[2])
                ding()
                done = True
            if parts[1][:6] == "second":
                timeOff = time.time() + int(parts[0])
                ding()
                done = True
            if parts[0][:4].lower() == "stop":
                timeOff = time.time() + infinity
                ding()
                done = True
        except:
            pass
        if not done:
            textIsOrdinateur = False
            try:
                textIsOrdinateur = transcribedText["text"].split()[0].lower()[:7] == "compute"
            except:
                pass
            if textIsOrdinateur:
                valid = time.time()
                ding()
            else:
                if time.time() - valid < 20:
                    if len(transcribedText["text"]) > 2:
                        discuss(transcribedText["text"])
                    valid = 0

# Start recording
p = pyaudio.PyAudio()
stream = p.open(format = frmat, channels = channels, rate = rate, input = True, frames_per_buffer = chunk)
samplesize = p.get_sample_size(frmat)

# timer
infinity = 100000000
timeOff = time.time() + infinity

while True:
    if time.time() > timeOff:
        ding()
        timeOff = time.time() + infinity
    data = stream.read(chunk, exception_on_overflow = False)
    npdata = (np.array(array.array('h', data)).astype(np.float32) / 32768)
    is_speech = vad.is_speech(data, rate)
    if is_speech and (time.time() - valid < 20 or len(speechFrames) < 15):
        speechFrames.append(npdata)
    else:
        if speechFrames != []:
            if len(speechFrames) > 10:
                stream.stop_stream()
                analyseSpeech(np.concatenate(speechFrames))
                stream.start_stream()
            speechFrames = []

