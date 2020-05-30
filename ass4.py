import face_recognition
import os

image = face_recognition.load_image_file(f"image_0001.jpg")
realID = face_recognition.face_encodings(image)[0]

for filename in os.listdir(test_faces):
    image = face_recognition.load_image_file(f"{test_faces}/{filename}")
    face = face_recognition.face_encodings(image, model="cnn")
    results = face_recognition.compare_faces([realID], face, 0.6)
    if True in results:
        print("Access Granted" + " " + filename)
    else:
        print("Access Denied" + " " + filename)