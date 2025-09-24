import os
import numpy as np
import librosa
import tensorflow as tf
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
from flask import send_file

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Aktifkan CORS hanya untuk frontend React (lebih aman daripada global)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load model CNN yang sudah dilatih
MODEL_PATH = "models/model_alto_sopran.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Label (ubah sesuai kelas dataset kamu)
LABELS = ["kelas_1", "kelas_2", "kelas_3"]

# Konfigurasi koneksi ke database MySQL
db_config = {
    "host": "localhost",
    "user": "root",        # ganti sesuai user MySQL
    "password": "",        # ganti sesuai password MySQL
    "database": "db_suara"
}

# Fungsi koneksi database
def get_db_connection():
    return mysql.connector.connect(**db_config)

# Fungsi untuk ekstraksi fitur audio (MFCC)
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error extracting features:", e)
        return None


# ------------------ API REGISTER ------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    npm = data.get("npm")
    password = data.get("password")
    nama = data.get("nama")
    role = data.get("role", "anggota")   # default role anggota

    if not npm or not password or not nama:
        return jsonify({"error": "Semua field wajib diisi"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # cek apakah npm sudah ada
    cursor.execute("SELECT * FROM pengguna WHERE npm = %s", (npm,))
    existing_user = cursor.fetchone()
    if existing_user:
        cursor.close()
        conn.close()
        return jsonify({"error": "NPM sudah terdaftar"}), 400

    # hash password
    hashed_password = generate_password_hash(password)

    cursor.execute("INSERT INTO pengguna (npm, password, nama, role) VALUES (%s, %s, %s, %s)",
                   (npm, hashed_password, nama, role))
    conn.commit()

    cursor.close()
    conn.close()

    return jsonify({"message": "Register berhasil"}), 201


# ------------------ API LOGIN ------------------
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
            "role": user["role"]   # kirim role ke frontend
        })
    else:
        return jsonify({"error": "NPM atau Password salah"}), 401


# ------------------ API PREDIKSI ------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    features = extract_features(file_path)
    if features is None:
        os.remove(file_path)
        return jsonify({"error": "Feature extraction failed"}), 500

    features = np.expand_dims(features, axis=-1)   # channel
    features = np.expand_dims(features, axis=0)    # batch

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    os.remove(file_path)

    return jsonify({
        "predicted_class": LABELS[predicted_class],
        "confidence": confidence
    })

# ------------------ API UPLOAD REKAMAN (Hanya anggota) ------------------
@app.route("/upload_recording", methods=["POST"])
def upload_recording():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload"}), 400

    file = request.files["file"]
    npm = request.form.get("npm")
    tugas_id = request.form.get("tugas_id")

    if not npm or not tugas_id:
        return jsonify({"error": "NPM dan Tugas ID wajib dikirim"}), 400

    # cek apakah tugas masih aktif
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM tugas WHERE id = %s", (tugas_id,))
    tugas = cursor.fetchone()

    if not tugas:
        cursor.close()
        conn.close()
        return jsonify({"error": "Tugas tidak ditemukan"}), 404

    if tugas["deadline"] < datetime.now():
        cursor.close()
        conn.close()
        return jsonify({"error": "Tugas sudah berakhir"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # 🔹 Ekstraksi fitur & Prediksi suara
    features = extract_features(save_path)
    if features is None:
        os.remove(save_path)
        return jsonify({"error": "Ekstraksi fitur gagal"}), 500

    features = np.expand_dims(features, axis=-1)   # channel
    features = np.expand_dims(features, axis=0)    # batch

    prediction = model.predict(features)
    predicted_class = LABELS[np.argmax(prediction, axis=1)[0]]
    confidence = float(np.max(prediction))

    # simpan hasil ke DB
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO rekaman (npm, filename, upload_time, tugas_id, predicted_class, confidence)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (npm, filename, datetime.now(), tugas_id, predicted_class, confidence))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        "message": f"Rekaman untuk tugas {tugas_id} berhasil diupload",
        "predicted_class": predicted_class,
        "confidence": confidence
    }), 200



# ------------------ API LIST REKAMAN (Hanya pelatih) ------------------
@app.route("/recordings", methods=["GET"])
def get_recordings():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT r.id, r.npm, p.nama, r.filename, r.upload_time
        FROM rekaman r
        JOIN pengguna p ON r.npm = p.npm
        ORDER BY r.upload_time DESC
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # ------------------ API BUAT TUGAS (Pelatih) ------------------
@app.route("/tugas", methods=["POST"])
def buat_tugas():
    data = request.get_json()
    judul = data.get("judul")
    deskripsi = data.get("deskripsi")
    deadline = data.get("deadline")   # format: "2025-10-01 23:59:00"
    dibuat_oleh = data.get("npm")     # NPM pelatih

    if not judul or not deadline or not dibuat_oleh:
        return jsonify({"error": "Judul, deadline, dan NPM wajib diisi"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tugas (judul, deskripsi, deadline, dibuat_oleh)
        VALUES (%s, %s, %s, %s)
    """, (judul, deskripsi, deadline, dibuat_oleh))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Tugas berhasil dibuat"}), 201


# ------------------ API LIST TUGAS (Semua User) ------------------
@app.route("/tugas", methods=["GET"])
def list_tugas():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM tugas ORDER BY deadline DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Tambahkan status aktif/berakhir
    now = datetime.now()
    for tugas in rows:
        tugas["status"] = "aktif" if tugas["deadline"] > now else "berakhir"

    return jsonify(rows), 200


# ------------------ API UPLOAD REKAMAN UNTUK TUGAS ------------------
@app.route("/upload_recording", methods=["POST"])
def upload_recording():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload"}), 400

    file = request.files["file"]
    npm = request.form.get("npm")
    tugas_id = request.form.get("tugas_id")

    if not npm or not tugas_id:
        return jsonify({"error": "NPM dan Tugas ID wajib dikirim"}), 400

    # cek apakah tugas masih aktif
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM tugas WHERE id = %s", (tugas_id,))
    tugas = cursor.fetchone()

    if not tugas:
        cursor.close()
        conn.close()
        return jsonify({"error": "Tugas tidak ditemukan"}), 404

    if tugas["deadline"] < datetime.now():
        cursor.close()
        conn.close()
        return jsonify({"error": "Tugas sudah berakhir"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO rekaman (npm, filename, upload_time, tugas_id)
        VALUES (%s, %s, %s, %s)
    """, (npm, filename, datetime.now(), tugas_id))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": f"Rekaman untuk tugas {tugas_id} berhasil diupload"}), 200


# ------------------ API LIHAT REKAMAN TUGAS (Pelatih) ------------------
@app.route("/rekaman_tugas/<int:tugas_id>", methods=["GET"])
def rekaman_tugas(tugas_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT r.id, r.npm, p.nama, r.filename, r.upload_time,
               r.predicted_class, r.confidence
        FROM rekaman r
        JOIN pengguna p ON r.npm = p.npm
        WHERE r.tugas_id = %s
        ORDER BY r.upload_time DESC
    """, (tugas_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify(rows), 200

# ------------------ API DOWNLOAD HASIL PREDIKSI (Pelatih) ------------------
@app.route("/download_hasil/<int:tugas_id>", methods=["GET"])
def download_hasil(tugas_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT r.id, r.npm, p.nama, r.filename, r.upload_time,
               r.predicted_class, r.confidence, t.judul
        FROM rekaman r
        JOIN pengguna p ON r.npm = p.npm
        JOIN tugas t ON r.tugas_id = t.id
        WHERE r.tugas_id = %s
        ORDER BY r.upload_time DESC
    """, (tugas_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return jsonify({"error": "Belum ada rekaman untuk tugas ini"}), 404

    # Buat DataFrame lalu export ke Excel
    df = pd.DataFrame(rows)
    filepath = f"hasil_prediksi_tugas_{tugas_id}.xlsx"
    df.to_excel(filepath, index=False)

    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
