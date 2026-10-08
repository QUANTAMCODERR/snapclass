# SnapClass

SnapClass is an AI-powered classroom attendance system built with Python and Streamlit. The application supports attendance verification using both face recognition and voice biometrics, with the goal of reducing manual attendance effort while maintaining a more secure and automated process.

This project combines a browser-based classroom dashboard, database-backed student records, and biometric matching pipelines for both visual and audio identity verification.

## 1. Project Overview

The application is designed around a simple classroom flow:

1. A teacher creates or manages classroom subjects.
2. Students register with profile data such as name, face embedding, and optional voice embedding.
3. Teachers run AI attendance checks from uploaded classroom photos or recorded audio.
4. The system compares detected biometric vectors against stored enrollment vectors.
5. Attendance results are logged to a database and displayed in the dashboard.

The main application entry point is [app.py](app.py), which routes between the student and teacher interfaces.

## 2. Core Architecture

### 2.1 Frontend / UI Layer
The user interface is implemented with Streamlit, which makes it easy to create interactive classroom screens, dialogs, and attendance review panels.

Key UI files include:

- [app.py](app.py)
- [src/screens/student_screen.py](src/screens/student_screen.py)
- [src/screens/teacher_screen.py](src/screens/teacher_screen.py)
- [src/components/dialog_voice_attendance.py](src/components/dialog_voice_attendance.py)
- [src/components/dialog_attendance_result.py](src/components/dialog_attendance_result.py)

Streamlit is used not only for page layout and widgets, but also for state handling, spinner dialogs, dialogs for enrollment and attendance, and in-app AI processing workflows.

### 2.2 Data Layer
The app persists user and attendance records using Supabase.

Relevant database logic:

- [src/database/db.py](src/database/db.py)
- [src/database/config.py](src/database/config.py)

Student records can include:

- student identifier
- name
- face_embedding
- voice_embedding
- enrollment mappings to subjects
- attendance log entries

This design allows a student to be matched against a stored biometric profile during recognition and attendance verification.

### 2.3 Biometric Recognition Pipelines
The system performs two key biometric tasks:

- face recognition
- voice recognition

These are implemented in:

- [src/pipelines/face_pipelines.py](src/pipelines/face_pipelines.py)
- [src/pipelines/voice_pipeline.py](src/pipelines/voice_pipeline.py)

## 3. Face Recognition Pipeline (Technical Details)

The face recognition system uses the Dlib face recognition stack combined with a linear Support Vector Machine (SVM).

### 3.1 Model and library usage
The explicit ML model library function used here is:

```python
facerec = dlib.face_recognition_model_v1(
    face_recognition_models.face_recognition_model_location()
)
```

This is the key face embedding model. It loads a pretrained Dlib model that converts a detected face into a 128-dimensional facial descriptor.

The project loads:

```python
 detector = dlib.get_frontal_face_detector()
 shape_predictor = dlib.shape_predictor(
     face_recognition_models.pose_predictor_model_location()
 )
 facerec = dlib.face_recognition_model_v1(
     face_recognition_models.face_recognition_model_location()
 )
```

This sequence is the standard Dlib face pipeline:

1. detect faces in the image
2. locate facial landmarks
3. generate a consistent face descriptor vector

### 3.2 Face embedding generation
In [src/pipelines/face_pipelines.py](src/pipelines/face_pipelines.py), the function `get_face_embeddings(image_np)` does the following:

```python
def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)
```

For each detected face:

```python
shape = sp(image_np, face)
face_descriptor = facerec.compute_face_descriptor(
    image_np, shape, 1
)
encodings.append(np.array(face_descriptor))
```

The descriptor is a 128-element vector representing face geometry. This vector is stored in the database as `face_embedding` and later compared against new face embeddings.

### 3.3 Classification strategy
Once student embeddings are loaded from the database, the project trains a multiclass SVM:

```python
clf = SVC(
    kernel='linear',
    probability=True,
    class_weight='balanced'
)
clf.fit(x, y)
```

This is a linear-kernel classifier for student identity classification. It is trained on stored face embeddings and student IDs.

During recognition:

```python
predicted_id = int(clf.predict([encoding])[0])
```

and then comparison is performed with:

```python
best_match_score = np.linalg.norm(student_embedding - encoding)
```

The system keeps a threshold:

```python
resemblance_threshold = 0.6
```

If the Euclidean distance between the candidate and the live embedding is below the threshold, the student is considered a positive match.

### 3.4 Why this matters
This means the face recognition system is not a deep CNN classifier trained end-to-end; instead, it is a classic biometric descriptor pipeline built on Dlib landmark detection and descriptor extraction, followed by SVM classification for recognition.

That approach is lightweight, fast enough for classroom use, and well suited to small-to-medium student cohorts.

## 4. Voice Recognition Pipeline (Technical Details)

The voice pipeline uses the Resemblyzer library, which is designed for speaker embeddings and voice verification.

### 4.1 Voice encoder
The project loads a voice encoder:

```python
from resemblyzer import VoiceEncoder, preprocess_wav

@st.cache_resource
def load_voice_encoder():
    return VoiceEncoder()
```

The voice encoder converts a spoken utterance into a vector embedding that captures speaker identity characteristics.

### 4.2 Audio preprocessing
Before embedding generation, the system converts the audio input into a normalized waveform:

```python
audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
wav = preprocess_wav(audio)
embedding = encoder.embed_utterance(wav)
```

This uses:

- librosa for audio loading and waveform processing
- 16 kHz resampling to standardize input
- Resemblyzer preprocessing to normalize the signal for speaker embedding extraction

### 4.3 Speaker matching and bulk detection
For each candidate student embedding:

```python
similarity = np.dot(new_embedding, stored_embedding)
```

The system compares the similarity score against a threshold:

```python
threshold = 0.65
```

During bulk classroom audio analysis, the project splits the audio into segments:

```python
segments = librosa.effects.split(audio, top_db=30)
```

Each segment is run through the voice encoder and matched against enrolled student voice embeddings. If a segment matches a candidate above threshold, it is recorded as a likely presence.

## 5. Attendance Flow

### 5.1 Student registration flow
In the student registration logic in [src/screens/student_screen.py](src/screens/student_screen.py):

- camera input is captured
- face embeddings are extracted
- optional audio input is captured
- voice embeddings are generated if audio is available
- a new student profile is written to the database

The registration path calls:

```python
encodings = get_face_embeddings(img)
face_emb = encodings[0].tolist()
voice_emb = get_voice_embedding(audio_data.read())
response_data = create_student(new_name, face_embedding=face_emb, voice_embedding=voice_emb)
```

### 5.2 Teacher attendance flow
The teacher takes attendance via:

- subject selection
- optional photo upload or classroom image capture
- face analysis of all captured photos
- results comparison against enrolled students
- attendance logging in the database

The teacher screen also supports a voice attendance dialog implemented in [src/components/dialog_voice_attendance.py](src/components/dialog_voice_attendance.py).

There, audio is captured with:

```python
audio_data = st.audio_input("Record classroom audio")
```

Then the system passes it to:

```python
detected_scores = process_bulk_audio(
    audio_bytes,
    candidates_dict
)
```

The detected scores are mapped to each enrolled student and converted into present/absent statuses.

## 6. Data Model and Database Design

The app uses Supabase tables for student profiles and attendance records. The exact pattern is visible in database functions.

Student profiles are created through functions like:

```python
def create_student(new_name, face_embedding=None, voice_embedding=None):
    data = {
        'name': new_name,
        'face_embedding': face_embedding,
        'voice_embedding': voice_embedding,
    }
```

Attendance is stored as structured log entries, including:

- student_id
- subject_id
- timestamp
- is_present

This makes records queryable and suitable for teacher dashboards and analytics.

## 7. Project File Structure

```text
snapclass/
├── app.py
├── requirements.txt
├── readme.md
├── src/
│   ├── components/
│   │   ├── dialog_attendance_result.py
│   │   ├── dialog_voice_attendance.py
│   │   └── ...
│   ├── database/
│   │   ├── config.py
│   │   └── db.py
│   ├── pipelines/
│   │   ├── face_pipelines.py
│   │   └── voice_pipeline.py
│   ├── screens/
│   │   ├── home_screen.py
│   │   ├── student_screen.py
│   │   └── teacher_screen.py
│   └── ui/
│       └── base_layout.py
└── venv/
```

## 8. Dependencies and Libraries

The project uses the following core libraries:

- Streamlit — web UI
- NumPy — numeric vector manipulation and embeddings
- Pandas — attendance output tables
- Dlib — face detection, landmarks, and face descriptor generation
- face_recognition_models — pretrained Dlib face models
- scikit-learn — `SVC` classifier
- librosa — waveform loading and segmentation
- Resemblyzer — speaker embedding generation
- Supabase — cloud database integration
- Pillow — image processing
- segno — QR code generation
- bcrypt — password hashing
- face_recognition_models GitHub package — pretrained facial model assets

## 9. Installation and Setup

### 9.1 Prerequisites

- Python 3.10+ recommended
- Windows is the target OS for this project, but Linux/macOS may work with the same package structure.
- For Dlib, a C++ build toolchain may be required on Windows.

### 9.2 Create environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 9.3 Install dependencies

```bash
pip install -r requirements.txt
```

If there are platform-specific compilation issues with Dlib on Windows, install the required Visual Studio C++ build components before retrying the install.

### 9.4 Run the app

```bash
streamlit run app.py
```

The application opens in the browser and lets teachers and students log in, register, and take attendance.

## 10. Configuration Notes

This app depends on a Supabase project for storage and identity records. There may be a config file under the database package that contains connection information and service credentials. Ensure that:

- Supabase URL is properly configured
- API keys are valid
- the relevant subject and attendance tables exist in your project
- all student and attendance queries match the schema used by the application

## 11. ML / Biometric Design Summary

The project uses a hybrid biometric approach:

- Face recognition: Dlib descriptors + SVC classification
- Voice recognition: Resemblyzer embeddings + cosine/inner-product similarity matching

This hybrid method is useful in real classroom settings because:

- face recognition is strong for visual identity checks
- voice recognition adds an additional factor for attendance validation
- the system can maintain a student profile with both biometric vectors

## 12. Strengths of the Implementation

- lightweight and simple for academic prototypes
- uses pretrained embeddings instead of retraining large neural models
- robust enough for classroom attendance workflows
- modular pipeline separation between UI, storage, and ML logic
- easy to extend with improved models or additional biometric features

## 13. Limitations and Practical Considerations

- Dlib face recognition is sensitive to image quality, glare, occlusion, background noise, and angle variance.
- Voice matching can degrade if the audio is noisy or if the student speaks too softly.
- The attendance thresholds are tuned heuristically (`0.6` for face distance and `0.65` for voice similarity) and may need adjustment for a different student population.
- The system is designed for educational settings and may need additional calibrations for large classrooms or poor audio conditions.

## 14. Implementation Notes for Future Development

Potential future enhancements include:

- adding deep learning based face verification models
- improving audio preprocessing and noise suppression
- adding anti-spoofing checks for voice and photo inputs
- building a teacher dashboard with attendance analytics and export
- introducing confidence scores and manual override operations
- adding secure multi-factor verification for high-stakes attendance systems

## 15. Final Summary

SnapClass is a technically grounded AI attendance system that merges:

- Streamlit-based classroom UX
- Supabase-backed student and attendance data
- Dlib-based face embedding extraction
- Resemblyzer-based voice embeddings
- SVM-based recognition for face identity
- threshold-based matching for check-ins

The key machine learning function used here is the Dlib face descriptor model loader:

```python
dlib.face_recognition_model_v1(...)
```

This function is the essential component that turns a detected face into a compact biometric fingerprint that the system can compare against stored student profiles.

This makes SnapClass a practical and educational example of AI-powered attendance automation using classical biometric matching techniques in a full-stack Python application.
