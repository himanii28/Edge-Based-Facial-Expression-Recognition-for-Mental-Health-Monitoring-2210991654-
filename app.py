import cv2
import numpy as np
import tensorflow as tf
import base64
import os
from flask import Flask, request, jsonify, render_template
from data_pipeline import preprocess_frame

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_fer_model.h5')
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a base64-encoded JPEG frame, runs face detection + FER inference,
    returns emotion probabilities as JSON.

    Request body (JSON):
        { "frame": "data:image/jpeg;base64,/9j/..." }

    Response (JSON):
        {
            "faces": 1,
            "emotion": "happy",
            "probabilities": [0.05, 0.01, 0.03, 0.72, 0.10, 0.05, 0.04],
            "coords": [x, y, w, h]   # bounding box of first face
        }
    """
    data = request.get_json(force=True, silent=True)
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame provided'}), 400

    # Decode base64 -> numpy BGR image
    raw = data['frame']
    if ',' in raw:
        raw = raw.split(',', 1)[1]   # strip "data:image/jpeg;base64,"
    img_bytes = base64.b64decode(raw)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'faces': 0, 'emotion': 'none', 'probabilities': [0.0] * 7, 'coords': []})

    faces_tensors, face_coords = preprocess_frame(frame)

    if not faces_tensors:
        return jsonify({'faces': 0, 'emotion': 'none', 'probabilities': [0.0] * 7, 'coords': []})

    # Run inference on first detected face only
    tensor = faces_tensors[0]
    pred = model.predict(np.expand_dims(tensor, axis=0), verbose=0)[0]
    probs = [float(p) for p in pred]
    dominant_idx = int(np.argmax(pred))

    return jsonify({
        'faces': len(faces_tensors),
        'emotion': EMOTION_LABELS[dominant_idx],
        'probabilities': probs,
        'coords': [int(c) for c in face_coords[0]]
    })


@app.route('/healthz')
def health():
    return 'ok', 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Dashboard -> http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
