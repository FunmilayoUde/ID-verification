import cv2
import numpy as np
import face_recognition
import time

def perform_face_detection(known_image):
    # Encode the known image
    known_image_np = np.array(known_image, dtype= np.uint8)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Initialize the VideoCapture object for the laptop camera
    cap = cv2.VideoCapture(0)

    # Check if webcam capture is successful
    if not cap.isOpened():
        print("Error opening video capture device")
        exit()

    # Flag to indicate a confirmed match
    face_matched = False
    start_time = time.time()  # Record start time

    while True:
        # Capture frame-by-frame
        ret, img = cap.read()

        # Check if frame capture is successful
        if not ret:
            print("Error capturing frame")
            break

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Stop after 70 seconds
        if elapsed_time > 70:
            break

        # Convert the frame to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)[0]

        # Loop through each detected face
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare face encoding with the extracted face encoding
            match = face_recognition.compare_faces([known_face_encoding], face_encoding)

            # Display message and draw rectangle based on match
            color = (0, 255, 0) if match else (0, 0, 255)  # Green for match, red for no match
            text = "Face Matched" if match else "Face Not Matched"
            cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)

            # Update flag if a match is found
            if match:
                face_matched = True

        # Display the resulting frame
        cv2.imshow('Face Recognition', img)

        # Break the loop if a match is confirmed or 'q' is pressed
        if face_matched or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


