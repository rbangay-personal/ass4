import face_recognition
import os
import cv2

TEST_FACES = "test_faces"
TOLERANCE = 0.6
MODEL = "cnn"


image = face_recognition.load_image_file(f"image_0001")
realID = face_recognition.face_encodings(image)[0]

for filename in os.listdir(TEST_FACES):
    print(filename)
    image = face_recognition.load_image_file(f"{TEST_FACES}/{filename}")
    faces = face_recognition.face_encodings(image)
    image = cv2.cvtColor(image, cv2.COLOR_RBG2BRG)

    for face in faces:
        results = face_recognition.compare_faces(realID, face, TOLERANCE)
        if True in results:
            print("match found" + filename[results.index(True)])
