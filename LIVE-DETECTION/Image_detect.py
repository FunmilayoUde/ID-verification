import cv2
from PIL import Image
import pytesseract
import sqlite3
import os
import numpy as np

def create_database():
    conn = sqlite3.connect('extracted_info.db')
    cursor = conn.cursor()

    # Create a table to store the extracted text
    cursor.execute('''CREATE TABLE IF NOT EXISTS extracted_text (
                        id INTEGER PRIMARY KEY,
                        text TEXT
                    )''')
    
    return conn, cursor

def extract_info_with_opencv(image, cursor, conn):
    img_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform contour detection to find the bounding box of the photo
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Extract the photo from the bounding box
    photo = img[y:y+h, x:x+w]

    # Convert the photo to PIL format for saving
    photo_image = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    
    # Save the photo to a folder
    save_folder = 'extracted_photos'
    os.makedirs(save_folder, exist_ok=True)
    photo_filename = os.path.join(save_folder, 'extracted_photo.jpg')
    photo_image.save(photo_filename)

    # Perform OCR to extract text from the entire image
    text = pytesseract.image_to_string(img)

    # Store the extracted text in the database
    cursor.execute("INSERT INTO extracted_text (text) VALUES (?)", (text,))
    conn.commit()

    # Close the database connection
    conn.close()
    
    return text, photo_image
