import face_recognition
import os

TEST_FACES = "test_faces"
TOLERANCE = 0.6
MODEL = "cnn"


image = face_recognition.load_image_file(f"image_0001.jpg")
realID = face_recognition.face_encodings(image)

for filename in os.listdir(TEST_FACES):
    print(filename)
    image = face_recognition.load_image_file(f"{TEST_FACES}/{filename}")
    face = face_recognition.face_encodings(image, model=MODEL)
    results = face_recognition.compare_faces(realID, face)
    if True in results:
        print("match found" + filename[results.index(True)])
