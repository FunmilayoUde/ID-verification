import streamlit as st
from Image_detect import extract_info_with_opencv, create_database
import cv2
import numpy as np
import face_recognition
import time


def main():
    st.title('ID Card Matching')

    # File uploader for ID card image
    st.subheader('Upload ID Card Image')
    id_card_image = st.file_uploader('Upload an image', type=['jpg', 'png'])
    conn, cursor = create_database()

    if id_card_image:
        # Extract text and photo from the uploaded ID card using OpenCV
        extracted_text, extracted_photo = extract_info_with_opencv(id_card_image,cursor, conn)
        
        # Display the extracted text
        st.subheader('Extracted Text:')
        st.write(extracted_text)

        # Display the extracted photo
        #st.subheader('Extracted Photo:')
        #st.image(extracted_photo, caption='Extracted Photo', use_column_width=True)
        known_image_np = np.array(extracted_photo, dtype= np.uint8)
        known_face_encoding = face_recognition.face_encodings(known_image_np)[0]
        if st.button("Perform Face Recognition"):

            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            stop_button_pressed = st.button("Stop")

            while cap.isOpened() and not stop_button_pressed:
                ret, img = cap.read()
                #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(img)
                face_encodings = face_recognition.face_encodings(img, face_locations)
                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    match = face_recognition.compare_faces([known_face_encoding], face_encoding)[0]
                    color = (0, 255, 0) if match else (0, 0, 255)  # Green for match, red for no match
                    text = "Face Matched" if match else "Face Not Matched"
                    cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                # Break the loop if a match is confirmed or 'q' is pressed
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(img_rgb, channels="RGB")
                if cv2.waitKey(1) & 0xFF == ord('q')or stop_button_pressed:
                    break
            cap.release()
            cv2.destroyAllWindows()




        
if __name__ == "__main__":
    main()
