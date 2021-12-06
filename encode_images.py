import face_recognition as fr
import cv2
import pickle


def get_encoded_faces():
    # looks through the faces folder and encodes all the faces
    encoded = {}
    for dir_path, d_names, face_names in cv2.os.walk("./faces"):
        for f in face_names:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    # encoded is a dict of(name, image encoded)
    with open("encodings.pickle", "wb") as f:
        pickle.dump(encoded, f)


get_encoded_faces()
