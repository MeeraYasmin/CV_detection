import speech_recognition as sr

def speech_to_text():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Speak something...")
        try:
            # Listen to the audio input
            audio = recognizer.listen(source)

            # Convert speech to text using Google's Web Speech API
            text = recognizer.recognize_google(audio)
            print("You said:", text)
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError:
            print("Could not request results from the speech recognition service.")

# Run the speech-to-text function
speech_to_text()
