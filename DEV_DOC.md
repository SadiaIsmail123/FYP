# Dev Doc

## Overview
This project is a Flask app that:
- Provides auth (register/login/logout)
- Captures images from the browser
- Runs emotion detection on saved images
- Uses a MySQL database for users
- Trains a custom PyTorch emotion model from image folders

## Project Structure
- `app.py` - App factory and blueprint registration
- `routes/` - Flask blueprints
  - `routes/auth.py` - Login/register/logout routes
  - `routes/emotion_detection.py` - Capture + emotion inference
- `db/` - Database connection and schema
  - `db/connection.py` - MySQL connection pool
  - `db/schema.py` - Table creation
- `templates/` - HTML templates
- `static/` - Static assets and captured images
- `models/` - Saved PyTorch model weights
- `train_emotion.py` - Model training script
- `requirements.txt` - Python dependencies

## Requirements
- Python 3.9+ (recommended)
- MySQL running locally
- Python packages in `requirements.txt`

Install dependencies:
```
pip install -r requirements.txt
```

## Database Setup
Edit `db/connection.py` if your MySQL settings are different:
```
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "detection_db",
}
```

Tables are created automatically at app startup:
- `users` for authentication
- `persons` (currently unused in routes)

## Run the App
```
python app.py
```
Default routes:
- `/login`
- `/register`
- `/logout`
- `/` (main capture page)
- `/capture/save` (POST)

## Training Data Format
Place training images in:
```
training_data/
  angry/
  fear/
  happy/
  neutral/
  sad/
  surprise/
```
Folder names become class labels.

## Train the Model
Auto-detect all emotion folders:
```
python train_emotion.py --data-dir training_data --epochs 3
```
Train a subset:
```
python train_emotion.py --data-dir training_data --emotions angry,fear,happy --epochs 3
```

Output:
- Model saved to `models/emotion_multi.pt`

## Inference (Emotion Detection)
`routes/emotion_detection.py` loads the trained model and predicts the emotion
for each captured image. If the model file is missing, it returns an error:
```
Trained model not found. Train it first.
```

## Notes
- If you trained with only one folder, the model can only predict that class.
- For multi-class detection, include multiple emotion folders.
- Intel GPU usage is not configured by default; training uses CUDA if available,
  otherwise CPU.
