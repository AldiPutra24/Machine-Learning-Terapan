# Laporan Proyek Machine Learning - Aldi Putra Miftaqull Ullum

## Project Overview

Di tengah perkembangan teknologi informasi yang pesat, sistem rekomendasi memainkan peran penting dalam meningkatkan pengalaman pengguna di platform hiburan, terutama di layanan streaming film. Proyek ini bertujuan untuk membangun **sistem rekomendasi berbasis konten** yang mampu memberikan rekomendasi film yang relevan berdasarkan deskripsi singkat (overview) dari film yang sudah ditonton atau disukai oleh pengguna.

### Mengapa Proyek Ini Penting?

Sistem rekomendasi yang efektif dapat memberikan manfaat signifikan, di antaranya:
- **Peningkatan Pengalaman Pengguna**: Dengan memberikan rekomendasi yang sesuai, pengguna dapat dengan mudah menemukan konten yang mereka sukai tanpa harus melakukan pencarian manual.
- **Retensi Pengguna**: Rekomendasi yang relevan dapat mendorong pengguna untuk tetap menggunakan platform, meningkatkan waktu tonton dan loyalitas pengguna.
- **Peningkatan Pendapatan**: Dengan lebih banyak pengguna yang terlibat dan puas, platform dapat mengalami peningkatan pendapatan melalui langganan dan iklan.

**Referensi:**
- Banik, Rounak. "The Movies Dataset." Kaggle. Tersedia di [Kaggle: The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset).
- Aggarwal, Charu C. *"Recommender Systems: The Textbook."* Springer, 2016.

---

## Business Understanding

Pada bagian ini, kita akan menjelaskan proses klarifikasi masalah yang menjadi fokus proyek ini.

### Problem Statements
1. **Bagaimana cara memberikan rekomendasi film yang relevan kepada pengguna berdasarkan kesamaan deskripsi film yang mereka tonton?**
   - Masalah ini muncul karena banyaknya film dengan genre yang sama, sehingga sulit bagi pengguna untuk menemukan film yang benar-benar sesuai dengan preferensi mereka.
   
2. **Bagaimana cara menangani judul film yang memiliki banyak versi atau duplikasi, sehingga rekomendasi tetap unik?**
   - Judul yang sama dapat merujuk pada film yang berbeda atau versi berbeda, yang berpotensi menyebabkan kebingungan bagi pengguna.

### Goals
1. **Membangun sistem rekomendasi film** yang dapat memberikan daftar film-film serupa berdasarkan kesamaan konten dari deskripsi film.
2. **Menyaring judul-judul film yang duplikat** untuk memastikan sistem hanya memberikan rekomendasi unik kepada pengguna.

### Solution Approach
Untuk mencapai tujuan tersebut, dua pendekatan utama digunakan:
1. **Pendekatan pertama: TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Mengonversi deskripsi film menjadi representasi vektor. Ini memungkinkan sistem untuk menghitung seberapa mirip dua film berdasarkan teks deskripsinya.
   
2. **Pendekatan kedua: Cosine Similarity**
   - Mengukur kemiripan antara film satu dengan yang lain setelah teks diubah menjadi vektor, lalu menghasilkan rekomendasi berdasarkan tingkat kemiripan tersebut.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini diambil dari **Full MovieLens Dataset**, yang mencakup metadata untuk 45.000 film yang dirilis pada atau sebelum Juli 2017. Data ini sangat berguna untuk analisis dan pengembangan sistem rekomendasi.

### Informasi Dataset
- **Jumlah Data**: Terdapat 45.000 film dalam dataset ini.
- **Kondisi Data**: Dataset memiliki berbagai fitur, termasuk anggaran, pendapatan, tanggal rilis, bahasa, perusahaan produksi, negara, jumlah suara, dan rata-rata suara di TMDB. 
- **Ukuran Dataset**: Dataset mencakup 26 juta rating dari 270.000 pengguna untuk semua film, dengan rating yang diberikan dalam skala 1-5.

### Konten Dataset
Dataset terdiri dari beberapa file penting:
1. **movies_metadata.csv**: 
   - File utama yang berisi informasi mengenai 45.000 film, termasuk poster, latar belakang, anggaran, pendapatan, tanggal rilis, bahasa, dan perusahaan produksi.
   
2. **keywords.csv**: 
   - Berisi kata kunci plot film untuk film-film dalam dataset, tersedia dalam bentuk objek JSON yang dinyatakan sebagai string.
   
3. **credits.csv**: 
   - Informasi mengenai cast dan crew untuk semua film, juga tersedia dalam bentuk objek JSON yang dinyatakan sebagai string.
   
4. **links.csv**: 
   - Mengandung ID TMDB dan IMDB dari semua film dalam dataset.
   
5. **ratings_small.csv**: 
   - Subset dari 100.000 rating yang diberikan oleh 700 pengguna untuk 9.000 film.

### Sumber Data
Dataset ini merupakan hasil pengumpulan data dari **TMDB** dan **GroupLens**. Detail film, kredit, dan kata kunci telah dikumpulkan dari TMDB Open API. Data rating diperoleh dari situs resmi GroupLens. 

### Variabel Penjelasan
- **title**: Judul film yang digunakan sebagai identifikasi.
- **overview**: Sinopsis yang memberikan gambaran tentang konten film, sangat penting untuk analisis rekomendasi.
- **budget**: Anggaran film yang dapat berhubungan dengan kesuksesan dan popularitas film.
- **revenue**: Pendapatan yang dihasilkan oleh film.
- **release_date**: Tanggal rilis film, relevan untuk analisis tren waktu.
- **keywords**: Kata kunci yang berkaitan dengan plot film, yang dapat membantu dalam analisis konten.

Data ini dapat diakses melalui tautan berikut: [Kaggle: The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset).

---

## Data Preparation

Proses persiapan data sangat penting untuk memastikan bahwa model dapat bekerja dengan efisien. Berikut adalah tahapan yang dilakukan:

1. **Mengisi Data Kosong (Missing Values)**
   - Pada kolom 'overview', seluruh nilai yang kosong digantikan dengan string kosong (`""`). Hal ini dilakukan agar algoritma TF-IDF dapat bekerja tanpa error.
   
2. **Menghapus Duplikasi Judul**
   - Film yang memiliki judul yang sama dihapus, kecuali satu versi terakhir. Ini bertujuan untuk memastikan sistem tidak memberikan rekomendasi yang sama berulang kali kepada pengguna.

3. **Pembersihan Data**
   - Membuang karakter khusus dan merapikan format teks agar lebih konsisten. Ini membantu meningkatkan akurasi model dalam menganalisis teks.

Proses persiapan data ini memastikan bahwa sistem dapat berjalan secara efisien dan akurat, serta meminimalkan potensi kesalahan dalam analisis.

---

## Modeling

Pada tahap pemodelan, kami menggunakan pendekatan **content-based filtering** yang memanfaatkan teks deskripsi film untuk merekomendasikan film yang mirip. Dua algoritma utama yang digunakan adalah:

1. **TF-IDF Vectorizer**
   - Algoritma ini mengubah deskripsi film menjadi representasi numerik, memungkinkan setiap film diwakili dalam bentuk vektor yang merefleksikan fitur penting dari teks.

2. **Cosine Similarity**
   - Setelah teks diubah menjadi vektor, algoritma ini digunakan untuk menghitung kemiripan antara vektor deskripsi film yang satu dengan yang lain. Berdasarkan hasil kemiripan ini, sistem dapat memberikan rekomendasi film yang paling mirip dengan film pilihan pengguna.

### Output
Sistem ini menghasilkan **Top-N Recommendation**, yaitu daftar 10 film paling mirip yang direkomendasikan berdasarkan film yang dipilih oleh pengguna. Sebagai contoh, jika pengguna memilih film "Cinderella", sistem akan menampilkan film-film lain yang memiliki deskripsi mirip dengan "Cinderella."

### Kelebihan dan Kekurangan
- **Kelebihan**: Pendekatan berbasis konten seperti ini bekerja dengan baik meskipun pengguna belum memiliki banyak riwayat tontonan, karena rekomendasi dibuat berdasarkan karakteristik film itu sendiri.
- **Kekurangan**: Sistem tidak mempertimbangkan faktor eksternal seperti preferensi pengguna, rating, atau interaksi pengguna lain, yang dapat membatasi kualitas rekomendasi dalam beberapa kasus.

---

## Evaluation

Metrik evaluasi yang digunakan dalam proyek ini adalah **Cosine Similarity**, yang mengukur tingkat kemiripan antara film satu dengan yang lain berdasarkan sinopsis film. Semakin tinggi nilai cosine similarity, semakin mirip dua film tersebut.

### Hasil Evaluasi
Proyek ini berhasil menghasilkan rekomendasi yang relevan untuk pengguna berdasarkan deskripsi film. Sistem yang dibangun dapat memberikan rekomendasi 10 film teratas yang mirip dengan film yang dipilih pengguna, serta mengatasi masalah duplikasi film dengan menjaga hanya satu versi film yang dipertimbangkan.

---

## Kesimpulan
Dengan menggunakan metode berbasis konten, sistem rekomendasi yang dibangun mampu membantu pengguna dalam menemukan film yang sesuai dengan preferensi mereka. Pengembangan lebih lanjut dapat mencakup integrasi data tambahan seperti rating pengguna atau metadata lainnya untuk meningkatkan akurasi dan relevansi rekomendasi. Proyek ini menunjukkan bahwa meskipun ada tantangan dalam pengelolaan data yang hilang dan duplikat, pendekatan yang tepat dapat menghasilkan solusi yang efektif dan efisien dalam sistem rekomendasi film.
