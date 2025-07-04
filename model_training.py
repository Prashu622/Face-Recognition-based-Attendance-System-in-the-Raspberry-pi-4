import os
from imutils import paths
import face_recognition
import pickle
import cv2
import gc  # Garbage collector

print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))
knownEncodings = []
knownNames = []

# Process images one by one, with resizing and memory cleanup
for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]

    # Load and resize image (smaller = faster and less memory)
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] skipping unreadable image: {imagePath}")
        continue

    image = cv2.resize(image, (500, int(image.shape[0] * (500 / image.shape[1]))))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use HOG model (lightweight; CNN will overload the Pi)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

    # Free memory manually
    del image, rgb, boxes, encodings
    gc.collect()

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Training complete. Encodings saved to 'encodings.pickle'")
