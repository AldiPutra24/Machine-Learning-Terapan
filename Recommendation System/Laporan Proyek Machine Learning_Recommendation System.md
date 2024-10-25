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
Kami akan menggunakan pendekatan **Content-based Filtering**, di mana **cosine similarity** digunakan sebagai metode untuk mengukur kesamaan antara film berdasarkan fitur-fitur yang ada dalam dataset.

---

## Data Understanding

Dataset yang digunakan dalam proyek adalah **movies_metadata.csv**, dataset ini diambil dari **Full MovieLens Dataset**, yang mencakup metadata untuk 45.000 film yang dirilis pada atau sebelum Juli 2017. Data ini sangat berguna untuk analisis dan pengembangan sistem rekomendasi.

### Informasi Dataset
- **Jumlah Data**: Terdapat 45.000 film dalam dataset ini.
- **Kondisi Data**: 
      - Dataset memiliki berbagai fitur, termasuk anggaran, pendapatan, tanggal rilis, bahasa, perusahaan produksi, negara, jumlah suara, dan rata-rata suara di TMDB. 
      - Missing value: Pada kolom overview terdapat 52 missing values, sedangkan pada kolom genre, 27 missing values. Pada kolom overview saja missing values yang dihapus untuk memastikan model dapat bekerja tanpa masalah.
- **Ukuran Dataset**: Dataset mencakup 26 juta rating dari 270.000 pengguna untuk semua film, dengan rating yang diberikan dalam skala 1-5.

### Konten Dataset
**movies_metadata.csv**: 
- File utama yang berisi informasi mengenai 45.000 film, termasuk poster, latar belakang, anggaran, pendapatan, tanggal rilis, bahasa, dan perusahaan produksi.

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

Proses persiapan data sangat penting untuk memastikan bahwa model dapat bekerja dengan efisien dan akurat. Berikut adalah tahapan yang dilakukan dalam proses ini:

1. **Mengisi Data Kosong (Missing Values)**
 - Pada kolom overview, seluruh nilai yang kosong digantikan dengan string kosong (""). Hal ini dilakukan agar algoritma TF-IDF dapat bekerja tanpa mengeluarkan error.
 - Jumlah Missing Values: Terdapat total 0 missing values pada kolom overview setelah pengisian ini.

2. **Menghapus Duplikasi Judul** 
 - Film yang memiliki judul yang sama dihapus, kecuali satu versi terakhir. Ini bertujuan untuk memastikan sistem tidak memberikan rekomendasi yang sama berulang kali kepada pengguna.
 - Jumlah Duplikasi Judul: Terdapat total 5 duplikasi yang dihapus.

3. **Pembersihan Data** 
 Membuang karakter khusus dan merapikan format teks agar lebih konsisten. Proses ini membantu meningkatkan akurasi model dalam menganalisis teks.

4. **Penggunaan TF-IDF**
 TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang digunakan untuk mengubah teks deskripsi film menjadi representasi numerik. TF-IDF memberikan bobot pada kata-kata yang penting dalam deskripsi dengan mempertimbangkan frekuensi kemunculannya dalam dokumen dan koleksi dokumen secara keseluruhan. Ini memungkinkan model untuk memahami konteks dan relevansi kata-kata dalam deskripsi film.

Proses persiapan data ini memastikan bahwa sistem dapat berjalan dengan efisien dan akurat, serta meminimalkan potensi kesalahan dalam analisis.

---

## Modeling

Pada tahap pemodelan, kami menggunakan pendekatan **content-based filtering** yang memanfaatkan teks deskripsi film untuk merekomendasikan film yang mirip. Dua algoritma utama yang digunakan adalah:

**Cosine Similarity**

 - Setelah teks diubah menjadi vektor menggunakan TF-IDF, algoritma ini digunakan untuk menghitung kemiripan antara vektor deskripsi film yang satu dengan yang lain.
 - Cara Kerja:
   - Cosine similarity mengukur sudut antara dua vektor dalam ruang multidimensi, yang memberikan nilai antara -1 dan 1. Semakin mendekati 1, semakin mirip dua film tersebut.
 - Parameter yang Digunakan:
   - Tidak ada parameter khusus, karena cosine similarity adalah fungsi yang menghitung kemiripan langsung dari vektor yang dihasilkan.

### Output
Sistem ini menghasilkan **Top-N Recommendation**, yaitu daftar 10 film paling mirip yang direkomendasikan berdasarkan film yang dipilih oleh pengguna. Sebagai contoh, jika pengguna memilih film "Batman Forever", sistem akan menampilkan film-film lain yang memiliki deskripsi mirip dengan "Batman Forever"

| **Judul**                                             | **Skor**   |
|------------------------------------------------------|------------|
| The Dark Knight Rises                                | 0.316211   |
| Batman: Bad Blood                                    | 0.254348   |
| Batman: Return of the Caped Crusaders               | 0.241174   |
| The Dark Knight                                      | 0.230327   |
| Batman: The Dark Knight Returns, Part 1             | 0.228748   |
| Batman: Mask of the Phantasm                         | 0.224106   |
| Batman Begins                                        | 0.211240   |
| Batman Unmasked: The Psychology of the Dark Knight   | 0.206121   |
| Batman Beyond: The Movie                             | 0.196737   |
| Batman & Bill                                        | 0.194599   |

### Kelebihan dan Kekurangan
- **Kelebihan**: Pendekatan berbasis konten seperti ini bekerja dengan baik meskipun pengguna belum memiliki banyak riwayat tontonan, karena rekomendasi dibuat berdasarkan karakteristik film itu sendiri.
- **Kekurangan**: Sistem tidak mempertimbangkan faktor eksternal seperti preferensi pengguna, rating, atau interaksi pengguna lain, yang dapat membatasi kualitas rekomendasi dalam beberapa kasus.

---

## Evaluation
Metrik evaluasi yang digunakan dalam proyek ini adalah Precision, yang mengukur seberapa banyak rekomendasi yang diberikan sistem relevan terhadap preferensi pengguna.

Precision adalah metrik yang digunakan untuk mengukur seberapa banyak rekomendasi yang relevan yang diberikan oleh sistem dari total rekomendasi yang dihasilkan. Rumusnya adalah sebagai berikut:

\[ 
\text{Precision} = \frac{\text{Jumlah Rekomendasi Relevan}}{\text{Jumlah Rekomendasi yang Diberikan}} 
\]

### Penjelasan:
- **Jumlah Rekomendasi Relevan:** Jumlah item yang direkomendasikan oleh sistem dan dianggap relevan oleh pengguna.
- **Jumlah Rekomendasi yang Diberikan:** Total item yang direkomendasikan oleh sistem.

### Hasil Evaluasi
Proyek ini berhasil menghasilkan rekomendasi yang relevan untuk pengguna berdasarkan deskripsi film. Untuk model content-based filtering ini, metrik precision dapat dihitung sebagai berikut:

| **Parameter**                       | **Nilai**  |
|-------------------------------------|------------|
| Jumlah Rekomendasi yang Relevan     | 8          |
| Jumlah Item yang Direkomendasikan    | 10         |
| **Precision**                       | **0.8**    |

\[
\text{Precision} = \frac{\text{Jumlah Rekomendasi yang Relevan}}{\text{Jumlah Item yang Direkomendasikan}} = \frac{8}{10} = 0.8 \quad (80\%) 
\]

---

## Kesimpulan
Dengan menggunakan metode berbasis konten, sistem rekomendasi yang dibangun mampu membantu pengguna dalam menemukan film yang sesuai dengan preferensi mereka. Pengembangan lebih lanjut dapat mencakup integrasi data tambahan seperti rating pengguna atau metadata lainnya untuk meningkatkan akurasi dan relevansi rekomendasi. Proyek ini menunjukkan bahwa meskipun ada tantangan dalam pengelolaan data yang hilang dan duplikat, pendekatan yang tepat dapat menghasilkan solusi yang efektif dan efisien dalam sistem rekomendasi film.
