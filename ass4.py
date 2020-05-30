import face_recognition
import os

def frange(start, stop, step):
    ''' "range()" like function which accept float type'''
    i = start
    while i < stop:
        yield i
        i += step

TEST_FACES = "test_faces"

true_positives = 0
false_positives = 0

image = face_recognition.load_image_file(f"image_0001.jpg")
realID = face_recognition.face_encodings(image)[0]
tolerance = frange(0, 1, 0.1)
for value in tolerance:
    for filename in os.listdir(TEST_FACES):
        print(filename)
        image = face_recognition.load_image_file(f"{TEST_FACES}/{filename}")
        if len(face_recognition.face_encodings(image, model="cnn")) != 0:
            face = face_recognition.face_encodings(image, model="cnn")[0]
            results = face_recognition.compare_faces([realID], face, tolerance)
        else:
            continue

        if True in results:
            if "real" in filename:
                true_positives += 1
            else:
                false_positives += 1
            # print("Access Granted" + " " + filename)
        else:
            # print("Access Denied" + " " + filename)
    print("true: " + str(true_positives) + '\n' + "false: " + str(false_positives))

