import cv2
import face_recognition
import numpy as np

# Load and encode the reference image
reference_img = face_recognition.load_image_file('Image/devansh.jpg')
reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
reference_img = cv2.resize(reference_img, (0, 0), None, 0.25, 0.25)
reference_face_encoding = face_recognition.face_encodings(reference_img)[0]

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Resize and convert the frame to RGB for face recognition
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Apply cartoonification on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon_frame = cv2.bitwise_and(color, color, mask=edges)

    # Process each detected face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([reference_face_encoding], face_encoding)
        face_distance = face_recognition.face_distance([reference_face_encoding], face_encoding)
        
        # Scale back up face locations since the frame was scaled down for face recognition
        top, right, bottom, left = [v * 4 for v in face_location]

        # Draw rectangle and label around the detected face on the cartoonified frame
        if matches[0]:
            cv2.rectangle(cartoon_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(cartoon_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(cartoon_frame, 'Devansh', (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.rectangle(cartoon_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(cartoon_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(cartoon_frame, 'Unknown', (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    # Display the final cartoonified frame with face recognition
    cv2.imshow('Face Recognition & Cartoonification', cartoon_frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
