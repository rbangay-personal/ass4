import face_recognition
import os



TEST_FACES = "test_faces"

true_positives = 0
false_positives = 0

image = face_recognition.load_image_file(f"image_0001.jpg")
realID = face_recognition.face_encodings(image)[0]
tolerance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for value in tolerance:
    print(value)
    for filename in os.listdir(TEST_FACES):
        image = face_recognition.load_image_file(f"{TEST_FACES}/{filename}")
        if len(face_recognition.face_encodings(image, model="cnn")) != 0:
            face = face_recognition.face_encodings(image, model="cnn")[0]
            results = face_recognition.compare_faces([realID], face, value)
        else:
            continue

        if True in results:
            if "real" in filename:
                true_positives += 1
            else:
                false_positives += 1
            # print("Access Granted" + " " + filename)
    print("true: " + str(true_positives) + '\n' + "false: " + str(false_positives))
    true_positives = 0
    false_positives = 0

