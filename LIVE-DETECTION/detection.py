import cv2
import sqlite3
import numpy as np

def create_database():
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    # Create table to store face images if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS face_images
                      (id INTEGER PRIMARY KEY, image BLOB)''')

    conn.commit()
    conn.close()

def store_face_image(image):
    # Convert the face image to binary format
    retval, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    # Connect to SQLite database
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    # Insert the face image into the database
    cursor.execute("INSERT INTO face_images (image) VALUES (?)", (sqlite3.Binary(image_bytes),))

    # Commit changes and close connection
    conn.commit()
    conn.close()

def extract_face_from_id_card(image_path):
    # Load image and detect face
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Assuming there's only one face in the image
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]
        return face_image
    else:
        return None

# Create the database if it doesn't exist
create_database()

# Example usage:
image_path = r'C:\Users\Laptop\Downloads\OCRA\LIVE-DETECTION\testcard.jpeg'
face_image = extract_face_from_id_card(image_path)

if face_image is not None:
    # Store the extracted face image in the database
    store_face_image(face_image)
    print("Face image stored in database.")
else:
    print("No face detected in the provided image.")
