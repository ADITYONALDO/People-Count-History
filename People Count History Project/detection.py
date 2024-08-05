import cv2
import numpy as np
import sqlite3
import os
import re
from datetime import datetime, timedelta

# Initialize database
conn = sqlite3.connect('people_detection.db')
c = conn.cursor()

# Create a single table for all entries
c.execute('''
    CREATE TABLE IF NOT EXISTS visits (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT,
        LastVisitTimestamp TEXT,
        NewVisitTimestamp TEXT
    )
''')
conn.commit()

# Load known faces
known_face_encodings = []
known_face_names = []

# Load LBP cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_frontalface.xml')

def normalize_name(name):
    """Remove numeric suffix and capitalize the first letter."""
    normalized_name = re.sub(r'\d+$', '', name).capitalize()
    return normalized_name

def load_known_faces(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_image = gray[y:y+h, x:x+w]
                face_encoding = cv2.resize(face_image, (100, 100)).flatten()
                known_face_encodings.append(face_encoding)
                normalized_name = normalize_name(os.path.splitext(filename)[0])
                known_face_names.append(normalized_name)

load_known_faces("known_faces")

# Function to detect and recognize faces
def detect_and_recognize(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_data = []
    
    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]
        face_encoding = cv2.resize(face_image, (100, 100)).flatten()
        name = "Unknown"
        
        min_distance = float('inf')
        for known_face, known_name in zip(known_face_encodings, known_face_names):
            distance = cv2.norm(known_face, face_encoding, cv2.NORM_L2)
            if distance < min_distance:
                min_distance = distance
                name = known_name
        
        # Set a distance threshold for recognizing faces
        threshold = 5200
        if min_distance > threshold:
            name = "Unknown"
        
        # Include timestamp for each detection
        timestamp = datetime.now()
        face_data.append((x, y, w, h, name, timestamp))
    
    return face_data

def update_or_insert_visit(name, timestamp):
    today = timestamp.strftime("%d/%m/%Y")
    formatted_timestamp = timestamp.strftime("%d/%m/%Y %I:%M %p")  # 12-hour format with AM/PM

    try:
        # Check if there is an existing entry for today and the given name
        c.execute('''
            SELECT ID, LastVisitTimestamp, NewVisitTimestamp
            FROM visits
            WHERE Name = ? AND LastVisitTimestamp LIKE ?
        ''', (name, f'{today}%'))
        
        result = c.fetchone()
        if result:
            visit_id, last_visited, new_visited = result
            last_visited_time = datetime.strptime(last_visited, "%d/%m/%Y %I:%M %p")
            current_time = datetime.strptime(formatted_timestamp, "%d/%m/%Y %I:%M %p")
            
            # Update NewVisitedTimestamp only if current time is greater than last visited time or new visited time
            if new_visited is None and current_time > last_visited_time:
                c.execute('''
                    UPDATE visits
                    SET NewVisitTimestamp = ?
                    WHERE ID = ?
                ''', (formatted_timestamp, visit_id))
            elif new_visited is not None and current_time > datetime.strptime(new_visited, "%d/%m/%Y %I:%M %p"):
                c.execute('''
                    UPDATE visits
                    SET NewVisitTimestamp = ?
                    WHERE ID = ?
                ''', (formatted_timestamp, visit_id))
        else:
            # Insert a new entry if none exists for that name on the current day
            c.execute('''
                INSERT INTO visits (Name, LastVisitTimestamp, NewVisitTimestamp)
                VALUES (?, ?, NULL)
            ''', (name, formatted_timestamp))
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

# Start video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

detected_names = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    face_data = detect_and_recognize(frame)
    
    for (x, y, w, h, name, timestamp) in face_data:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{name} - {timestamp.strftime('%d/%m/%Y %I:%M %p')}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        if name != "Unknown":
            # Check if the name has been detected within the last minute
            if name not in detected_names or timestamp - detected_names[name] > timedelta(minutes=1):
                update_or_insert_visit(name, timestamp)
                detected_names[name] = timestamp
    
    cv2.putText(frame, f"Detected Faces: {len(face_data)}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 250), 2)
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
