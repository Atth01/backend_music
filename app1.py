import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import mysql.connector
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------- CONFIG -------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

MODEL_PATH = "models/model_deteksi.h5"
model = tf.keras.models.load_model(MODEL_PATH)
LABELS = ["Alto", "Sopran"]

ALLOWED_EXTENSIONS = {'wav'}

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "db_suara"
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------- FEATURE EXTRACTION -------------------
SR = 22050
N_MFCC = 40
MAX_PAD_LEN = 174

def extract_features(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sample_rate = librosa.load(file_path, sr=SR)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)

        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.T
    except Exception as e:
        print("Error extracting features:", e)
        return None

# ------------------ REGISTER ------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    npm = data.get("npm")
    password = data.get("password")
    nama = data.get("nama")
    role = data.get("role", "anggota")

    if not npm or not password or not nama:
        return jsonify({"error": "Semua field wajib diisi"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM pengguna WHERE npm = %s", (npm,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({"error": "NPM sudah terdaftar"}), 400

    hashed_password = generate_password_hash(password)
    cursor.execute(
        "INSERT INTO pengguna (npm, password, nama, role) VALUES (%s, %s, %s, %s)",
        (npm, hashed_password, nama, role)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Register berhasil"}), 201

# ------------------ LOGIN ------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    npm = data.get("npm")
    password = data.get("password")

    if not npm or not password:
        return jsonify({"error": "NPM dan Password wajib diisi"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM pengguna WHERE npm = %s", (npm,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and check_password_hash(user["password"], password):
        return jsonify({
            "message": "Login berhasil",
            "nama": user["nama"],
            "npm": user["npm"],
            "role": user["role"]
        })
    return jsonify({"error": "NPM atau Password salah"}), 401

# ------------------ UPLOAD & PREDIKSI ------------------
@app.route('/upload_task_recording', methods=['POST'])
def upload_task_recording():
    try:
        file = request.files.get('file')
        npm = request.form.get('npm')

        if not file or not npm:
            return jsonify({"error": "File dan npm wajib diisi"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Hanya file .wav yang diperbolehkan"}), 400

        # Simpan file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Ekstraksi fitur
        features = extract_features(file_path)
        if features is None:
            return jsonify({"error": "Ekstraksi fitur gagal"}), 500

        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        predicted_class = LABELS[np.argmax(prediction, axis=1)[0]]
        confidence = float(np.max(prediction))

        # Simpan ke database
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO record (npm, filename, upload_time, predicted_class, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """
        values = (npm, filename, datetime.now(), predicted_class, confidence)
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()

        # Hapus file sementara
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "message": "Upload sukses",
            "npm": npm,
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ LIST ANGGOTA ------------------
@app.route("/anggota", methods=["GET"])
def list_anggota():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT npm, nama FROM pengguna WHERE role = 'anggota' ORDER BY nama ASC")
        rows = cursor.fetchall()
        return jsonify(rows), 200
    finally:
        cursor.close()
        conn.close()

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
