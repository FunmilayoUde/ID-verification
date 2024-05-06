import cv2
import os

def extract_info_from_id_card(image_path):
    image = cv2.imread(image_path)
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Face extraction
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        # Assuming there's only one face in the image
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]
        
        # Define the output folder path
        output_folder = 'extracted_images'
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Define the output image path
        output_image_path = os.path.join(output_folder, 'extracted_face.jpg')
        # Write the extracted face image to the output folder
        cv2.imwrite(output_image_path, face_image)
        print("Extracted image saved successfully.")
    else:
        print("No face found in the image.")

# Example usage:
image_path = r'C:\Users\Laptop\Downloads\OCRA\LIVE-DETECTION\testcard.jpeg'
extract_info_from_id_card(image_path)
