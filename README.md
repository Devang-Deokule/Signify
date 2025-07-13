✋🤟 Signify — Real-Time Sign Language to Speech & Text
Finalist project at Code Strom Hackathon 2025 | DIT University, Dehradun | Social Impact Theme
---
🚀 Overview
Signify is a real-time Sign Language Recognition System that bridges communication gaps for the deaf and mute community. It captures hand signs through a webcam, detects gestures for letters A–Z and special signs, and instantly translates them into text and speech — enabling seamless communication between signers and non-signers.
---
🎯 Problem Statement
Millions of deaf and mute individuals use sign language to express themselves, but most people around them can’t understand it. This communication barrier creates daily challenges. Signify tackles this problem by providing an accessible, real-time translation system to promote inclusivity.
---
🗝️ Key Features
Real-time hand landmark detection using MediaPipe
Custom-trained Random Forest Classifier on a self-built dataset
Instant text and text-to-speech output
Web-based system using Flask, accessible via any browser
Plans for mobile app integration with TensorFlow Lite
---
⚙️ Tech Stack
Languages & Libraries:
Python
OpenCV
MediaPipe
NumPy
scikit-learn
pyttsx3 (Text-to-Speech)
Flask (backend)
HTML, CSS, JavaScript (frontend)
Pickle (model serialization)
Tools:
VS Code (IDE)
Git & GitHub (version control)
---
📊 Dataset & Model
Custom Dataset: Captured using webcam and MediaPipe hand landmarks (21 points, flattened to 42 features per sample). Multiple samples per class for signs A–Z and special gestures.
Preprocessing: Normalized landmarks and labeled data points.
Model: RandomForestClassifier trained with scikit-learn and serialized as model.p.
Innovation: Entire dataset built in-house for relevance and accuracy without relying on pre-trained models.
---
🖥️ How It Works
Captures live webcam feed using OpenCV.
Extracts hand landmarks in real time with MediaPipe.
Normalizes and feeds landmarks into the trained model.
Displays the predicted sign on screen and generates speech output.
Accessible via a simple Flask web interface.
---
📱 Future Scope
Mobile App: Develop a responsive app with TensorFlow Lite + MediaPipe for on-device detection.
Two-Way Communication: Add voice/text input to generate sign language animations (reverse translation).
Continuous Learning: Integrate user feedback to improve prediction accuracy over time.
Regional Support: Expand to multiple regional sign languages.
---
💙 Social Impact
Signify was built under the Social Impact theme to empower the hearing and speech impaired community. It aims to reduce communication barriers and promote inclusivity in everyday life.
---
