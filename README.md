# Pengambilan Informasi
Tugas mata kuliah Temu Kembali Informasi. Tujuan dari tugas ini adalah untuk mempelajari dasar-dasar Pengambilan Informasai dan proses membangun indeks mesin pencari sederhana menggunakan pustaka Pyserini (antarmuka Python untuk Lucene/Anserini). Projek ini akan membandingkan bagaimana perbedaan hasil dari kueri pada metode Boolean dan BM25 secara sederhana.

## Setup Java
* Buka Environmental Variables
* Pastikan pada bagian System Variables terdapat JAVA_HOME dengan value: C:\Program Files\Java\jdk-21 (atau path instalasi ke jdk-21)
* Jika belum ada, tambahkan variabel baru bernama JAVA_HOME dengan value path ke jdk-21

## Setup
### 1. Kloning Repository
```
git clone https://github.com/lituldust/boolean-retrieval
cd boolean-retrieval
```
### 2. Membuat Virtual Environment
```
python -m venv .venv
.venv/scripts/activate
```
### 3. Install Package yang Dibutuhkan
```
pip install -r requirements.txt
```
