import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading  # 🔹 Multithreading to prevent lag

# Load trained model
model = tf.keras.models.load_model("best_model.h5")

# Define class labels
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Define custom messages for each emotion
emotion_responses = {
    "Angry": "Calm down, take a deep breath. Everything will be fine!",
    "Disgust": "Hmm, that doesn’t look good. What’s bothering you?",
    "Fear": "Don’t be afraid! You are stronger than you think.",
    "Happy": "Wow! You look really happy today! Keep smiling!",
    "Neutral": "You seem neutral. Thinking about something?",
    "Sad": "Hey, don’t be sad! I’m here to cheer you up!",
    "Surprise": "Oh! You look surprised! What happened?"
}

# Initialize pyttsx3 (Text-to-Speech)
engine = pyttsx3.init()

# Function to speak asynchronously
def speak_async(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait())).start()

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 🔹 Use DirectShow for better Windows performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # 🔹 Lower resolution increases FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0  # Normalize

        # Predict emotion
        prediction = model.predict(roi_gray, verbose=0)
        emotion_label = class_labels[np.argmax(prediction)]

        # Speak the emotion without blocking video
        speak_async(emotion_responses.get(emotion_label, "I see you!"))

        # Display emotion on frame
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Facial Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
