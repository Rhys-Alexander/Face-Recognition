import face_recognition as fr
import cv2
import numpy as np
import tkinter as tk
import math
import pickle


def classify_faces(img):
    # will find all of the faces in a given image and label them if it knows what they are
    with open("encodings.pickle", "rb") as f:
        faces = pickle.load(f)
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if name == "Unknown":
                colour = (0, 0, 0)
            else:
                colour = (0, 200, 0)

            # Draw a box around the face
            cv2.rectangle(
                img, (left - 20, top - 20), (right + 20, bottom + 20), colour, 2
            )

            # Draw a label with a name below the face
            cv2.rectangle(
                img,
                (left - 20, bottom - 15),
                (right + 20, bottom + 20),
                colour,
                cv2.FILLED,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2
            )

    return img, face_names


def display_image(img, message):
    # Display the resulting image
    while True:
        cv2.imshow(message, img)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break


def resize_image(img, area=0.0, window_h=0, window_w=0):
    h, w = img.shape[:2]
    root = tk.Tk()
    screen_h = root.winfo_screenheight()
    screen_w = root.winfo_screenwidth()

    if area != 0.0:
        vector = math.sqrt(area)
        window_h = screen_h * vector
        window_w = screen_w * vector

    if h > window_h or w > window_w:
        if h / window_h >= w / window_w:
            multiplier = window_h / h
        else:
            multiplier = window_w / w
        img = cv2.resize(img, (0, 0), fx=multiplier, fy=multiplier)

    return img


def get_image():
    # initialize the camera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 -> index of camera
    while True:
        s, frame = cam.read()
        cv2.imshow("Manual Select", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
    if s:  # frame captured without any errors
        return frame


def auto_verify():
    with open("encodings.pickle", "rb") as f:
        faces = pickle.load(f)
    known_face_names = list(faces.keys())
    known_face_encodings = list(faces.values())

    # initialize the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 -> index of camera
    process_this_frame = True
    face_found = False
    name = "Unknown"
    while True:
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_encodings = fr.face_encodings(rgb_small_frame)
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = fr.compare_faces(known_face_encodings, face_encoding)
                # use the known face with the smallest distance to the new face
                face_distances = fr.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    face_found = True

        process_this_frame = not process_this_frame

        if face_found:
            cap.release()
            break

    return name


def real_time_verify():
    with open("encodings.pickle", "rb") as f:
        faces = pickle.load(f)
    known_face_names = list(faces.keys())
    known_face_encodings = list(faces.values())

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_locations = []
    names = []
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = fr.face_locations(rgb_small_frame)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

            names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = fr.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = fr.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, names):
            if name == "Unknown":
                colour = (0, 0, 0)
            else:
                colour = (0, 200, 0)
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), colour, cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        # Display the resulting image
        resize_image(frame, area=0.25)
        cv2.imshow("Real Time Selection", frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()
    while names.count("Unknown") != 0:
        names.remove("Unknown")
    try:
        name = names[0]
        greeting = f"Welcome {name}"
        image = resize_image(frame, area=0.0625)
        display_image(image, greeting)
    except IndexError:
        print("No One Found")
        name = real_time_verify()
    cv2.destroyAllWindows()
    return name


def manual_verify(file_path=0):
    if file_path == 0:
        image, names = classify_faces(get_image())
        area = 0.0625
    else:
        image = cv2.imread(file_path)
        image, names = classify_faces(image)
        area = 0.5

    while names.count("Unknown") != 0:
        names.remove("Unknown")

    try:
        name = names[0]
        greeting = f"Welcome {name}"
        image = resize_image(image, area=area)
        display_image(image, greeting)
    except IndexError:
        print("No One Found")
        name = manual_verify(file_path)
    cv2.destroyAllWindows()
    return name
