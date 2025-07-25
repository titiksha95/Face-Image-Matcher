import cv2
import insightface
import numpy as np
import faiss
import pickle
import os
import shutil
from sklearn.preprocessing import normalize

# Load InsightFace model
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)  # CPU=-1, GPU=0

# Load dataset
embeddings = np.load("embeddings.npy")
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Normalize for cosine similarity
embeddings = normalize(embeddings, axis=1)

# Build FAISS cosine similarity index
index = faiss.IndexFlatIP(512)
index.add(embeddings)

# Webcam capture
cap = cv2.VideoCapture(0)
print("📷 Press 's' to capture image for search...")

frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Webcam - Press 's' to capture", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

# Extract embedding from webcam image
faces = model.get(frame)
if not faces:
    print("❌ No face detected in webcam image.")
    exit()

query_embedding = normalize(faces[0].embedding.reshape(1, -1), axis=1)

# Search all
D, I = index.search(query_embedding, len(filenames))
threshold = 0.3  # cosine similarity threshold (higher = more similar)

# Create folder for matched results
matched_dir = "matched_faces"
os.makedirs(matched_dir, exist_ok=True)

match_count = 0
for score, idx in zip(D[0], I[0]):
    if score < threshold:
        continue  # ignore poor matches

    img_file = filenames[idx].strip()
    img_path = os.path.join("event_photos", img_file)

    if not os.path.isfile(img_path):
        print(f"❌ File not found: {img_path}")
        continue

    # Copy matched image
    shutil.copy(img_path, os.path.join(matched_dir, img_file))

    # Show image (resized for display, not zoomed)
    img = cv2.imread(img_path)
    if img is not None:
        resized = cv2.resize(img, (600, 400))
        cv2.imshow(f"Match {match_count + 1}", resized)

    print(f"✅ Match {match_count + 1}: {img_file} - Similarity: {score:.4f}")
    match_count += 1

if match_count == 0:
    print("❌ No matches found above threshold.")
else:
    print(f"\n📁 {match_count} images saved to '{matched_dir}'")

cv2.waitKey(0)
cv2.destroyAllWindows()