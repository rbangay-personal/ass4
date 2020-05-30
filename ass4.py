import face_recognition
import os

TEST_FACES = "test_faces"

true_positives = 0
false_positives = 0

image = face_recognition.load_image_file(f"image_0001.jpg")
realID = face_recognition.face_encodings(image)[0]

for filename in os.listdir(TEST_FACES):
    image = face_recognition.load_image_file(f"{TEST_FACES}/{filename}")
    face = face_recognition.face_encodings(image, model="cnn")[0]
    results = face_recognition.compare_faces([realID], face, 0.6)
    if True in results:
        if "real" in filename:
            true_positives += 1
        else:
            false_positives += 1
        print("Access Granted" + " " + filename)
    else:
        print("Access Denied" + " " + filename)