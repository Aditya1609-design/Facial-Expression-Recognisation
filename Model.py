import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time
from collections import deque, Counter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the trained model
model = tf.keras.models.load_model("best_model.h5")
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
emoji_map = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨", "Happy": "😄",
    "Neutral": "😐", "Sad": "😢", "Surprise": "😲"
}
emotion_responses = {
    "Angry": "Calm down, take a deep breath. Everything will be fine!",
    "Disgust": "Hmm, that doesn’t look good. What’s bothering you?",
    "Fear": "Don’t be afraid! You are stronger than you think.",
    "Happy": "Wow! You look really happy today! Keep smiling!",
    "Neutral": "You seem neutral. Thinking about something?",
    "Sad": "Hey, don’t be sad! I’m here to cheer you up!",
    "Surprise": "Oh! You look surprised! What happened?"
}

engine = pyttsx3.init()
def speak_async(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait())).start()

# Live graph tracking
emotion_history = deque(maxlen=30)
trigger_buffer = deque(maxlen=5)

# Setup graph
emotion_counts = {label: 0 for label in class_labels}
plt.style.use('seaborn')
fig, ax = plt.subplots()
bar_container = ax.bar(class_labels, [0] * len(class_labels))
def update_graph(frame):
    counts = Counter(emotion_history)
    for i, label in enumerate(class_labels):
        bar_container[i].set_height(counts.get(label, 0))
    ax.set_ylim(0, max(5, max(counts.values(), default=1)))
    return bar_container
FuncAnimation(fig, update_graph, interval=1000)
threading.Thread(target=plt.show).start()

# Webcam setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 30)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_output.avi', fourcc, 20.0, (640, 480))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

trigger_responses = {
    "Sad": {
        "count": 3,
        "message": "You've been looking sad for a while. Try taking a short break or talking to a friend."
    }
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected_emotion = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0

        prediction = model.predict(roi_gray, verbose=0)[0]
        max_index = np.argmax(prediction)
        emotion_label = class_labels[max_index]
        confidence = float(prediction[max_index]) * 100
        emoji = emoji_map.get(emotion_label, "")

        emotion_history.append(emotion_label)
        trigger_buffer.append(emotion_label)
        detected_emotion = emotion_label

        label_display = f"{emotion_label} {emoji} ({confidence:.1f}%)"
        cv2.putText(frame_resized, label_display, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

        speak_async(emotion_responses.get(emotion_label, "I see you!"))
        break

    # Trigger checks
    for emotion, info in trigger_responses.items():
        if trigger_buffer.count(emotion) >= info["count"]:
            speak_async(info["message"])
            trigger_buffer.clear()

    out.write(frame_resized)
    cv2.imshow("Facial Emotion Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

