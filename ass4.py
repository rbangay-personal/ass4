import face_recognition
import os

TEST_FACES = "test_faces"
TOLERANCE = 0.6
MODEL = "cnn"


image = face_recognition.load_image_file(f"image_0001.jpg")
realID = face_recognition.face_encodings(image)[0]

for filename in os.listdir(TEST_FACES):
    image = face_recognition.load_image_file(f"{TEST_FACES}/{filename}")
    face = face_recognition.face_encodings(image)[0]
    results = face_recognition.compare_faces([realID], face, TOLERANCE)
    if True in results:
        print("match found" + " " + filename)