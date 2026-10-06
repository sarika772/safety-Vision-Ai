import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

voices = engine.getProperty("voices")

print("Number of voices:", len(voices))

for i, voice in enumerate(voices):
    print(i, voice.name)

def speak(message):
    print("VOICE ALERT:", message)
    engine.say(message)
    engine.runAndWait()