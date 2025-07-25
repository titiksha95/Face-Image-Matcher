import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import normalize
import insightface

# Initialize InsightFace
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)

image_folder = "event_photos"
embeddings = []
filenames = []

for file in os.listdir(image_folder):
    img_path = os.path.join(image_folder, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Could not read: {file}")
        continue

    faces = model.get(img)
    if not faces:
        print(f"😶 No face detected in: {file}")
        continue

    for i, face in enumerate(faces):
        emb = face.embedding
        embeddings.append(emb)
        # Store original filename + index if multiple faces in one image
        filenames.append(file)

print(f"🧠 Total embeddings extracted: {len(embeddings)}")

# Normalize and save
embeddings = normalize(np.array(embeddings), axis=1)
np.save("embeddings.npy", embeddings)

with open("filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print("✅ embeddings.npy and filenames.pkl saved.")
