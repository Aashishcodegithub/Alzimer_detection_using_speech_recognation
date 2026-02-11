# 🧠 Alzheimer Detection Using Speech Recognition

A cutting-edge AI project that uses **speech recognition and machine learning** to detect early signs of *Alzheimer’s disease* from voice recordings.

This repository combines speech preprocessing, feature extraction, and predictive modeling to build a non-invasive system for detecting cognitive decline from speech patterns — empowering early diagnosis with modern AI techniques.

---

## 🚀 Project Overview

Alzheimer’s is a progressive neurological disorder that impacts memory, behavior and communication. Speech carries subtle cues — like pauses, prosody, fluency and language use — that can signal cognitive impairment. By leveraging **AI and speech recognition**, this project aims to classify speech samples into Alzheimer’s vs. non-Alzheimer’s categories using deep learning and traditional ML models. :contentReference[oaicite:0]{index=0}

---

## 🧩 Features

- 🎙️ **Speech Recognition**
  - Transcribes audio recordings into text using an ASR engine.

- 🧠 **AI-Driven Detection**
  - Extracts acoustic and linguistic features from speech.
  - Trains ML/DL models to predict early cognitive signs.

- 📊 **Visual Results & Predictions**
  - Outputs probability scores and prediction results.
  - (Optional) Visualization of spectrograms/features.

- 🔧 Modular backend and frontend support for easy extension.

---

## 📁 Repo Structure

Alzimer_detection_using_speech_recognation/
│
├── Backend/ # AI models, speech processing & API logic
│ ├── app.py # Main backend application
│ ├── model/ # Trained ML/DL models
│ ├── utils/ # Helper functions (feature extraction, preprocessing)
│ └── requirements.txt # Backend dependencies
│
├── Frontend/ # User interface
│ ├── public/ # Static assets
│ ├── src/ # Frontend source code
│ └── package.json # Frontend dependencies
│
├── test/ # Sample audio files for testing
│
├── README.md # Project documentation
└── .gitignore # Ignored files
