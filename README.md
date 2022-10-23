# face recognition

_using the face_recognition lib and opencv in Python_

## encode_images.py

This will encode the faces belonging to the images in the faces folder and store them.

The faces will be named after the name of the image they come from.

## user_face_verification.py

### auto_verify()

This will look until it sees someone it recognises and then return their name.

### real_time_verify()

This will show a window of what and who it sees.

Press q to take the image that'll be processed.

### manual_verify()

This does the same as real_time_verify but without showing who in the window, it can also take a file path as a parameter if you want to work on an image.
