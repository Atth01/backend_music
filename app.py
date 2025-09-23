import os
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
import tensorflow as tf
from werkzeug.utils import secure_filename

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model CNN yang sudah dilatih
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Label (ubah sesuai kelas dataset kamu)
LABELS = ["kelas_1", "kelas_2", "kelas_3"]

# Fungsi untuk ekstraksi fitur audio (MFCC)
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Padding supaya ukuran seragam
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error extracting features:", e)
        return None

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Simpan file sementara
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Ekstraksi fitur
    features = extract_features(file_path)
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    # Ubah ke bentuk input model CNN
    features = np.expand_dims(features, axis=-1)   # tambahkan channel
    features = np.expand_dims(features, axis=0)    # batch dimension

    # Prediksi
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    # Hapus file setelah diproses
    os.remove(file_path)

    return jsonify({
        "predicted_class": LABELS[predicted_class],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
