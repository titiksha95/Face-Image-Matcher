# 🔍 Face Recognition & Matching System

A Python-based real-time face recognition tool that detects a face from a webcam and matches it against a dataset of photos (including group images) using deep learning and FAISS similarity search.

## 📸 Features

- Detect faces using [InsightFace](https://github.com/deepinsight/insightface)
- Support for group photos
- Generate and search facial embeddings
- Match captured face with closest faces in your dataset
- Uses FAISS for fast cosine similarity search
- Live webcam support for real-time capture

## 🛠 Technologies Used

- Python
- OpenCV
- InsightFace
- NumPy
- FAISS
- scikit-learn
- Pickle

## 📁 Project Structure

```
project/
│
├── face.py                     # Main matching script (webcam)
├── generate_embeddings.py      # Embedding generator (single face per image)
├── filenames.pkl               # Corresponding image names
├── event_photos/               # [save your images in folder named this]
├── matched_faces/              # Output folder for matched results
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/titiksha95/Face-Image-Matcher.git
cd Face-Image-Matcher
```

### 2. Set up virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Add your dataset

Place your images (even group photos) inside `event_photos/`

### 5. Generate embeddings

```bash
python generate_embeddings.py
```

### 6. Run the face matcher

```bash
python face.py
```

Press `s` to capture a frame from your webcam. It will match the detected face against your dataset and show the results.

## 📌 Notes

- `embeddings.npy` and `filenames.pkl` are auto-generated.
- `matched_faces/` will contain copies of matched images.
- For best results, use clear and front-facing images.


