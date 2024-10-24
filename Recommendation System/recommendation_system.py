# -*- coding: utf-8 -*-
"""Recommendation System.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XNOh6ooYqG5a_RrHqa31itH-rzFMwA_X
"""

# Mengimpor modul kagglehub untuk mengunduh dataset dari Kaggle
import kagglehub

# Mengunduh dataset terbaru "The Movies Dataset" dari Kaggle
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

# Menampilkan jalur tempat dataset disimpan
print("Path to dataset files:", path)

"""## Data Understanding

"""

# Mengimpor pandas untuk pemrosesan data dan beberapa modul dari scikit-learn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Membaca dataset 'movies_metadata.csv' dari direktori yang sudah diunduh
df = pd.read_csv('/root/.cache/kagglehub/datasets/rounakbanik/the-movies-dataset/versions/7/movies_metadata.csv')

# Melihat 5 baris pertama dari dataset
df.head()

# Menampilkan informasi dataset, termasuk jumlah kolom dan tipe data masing-masing kolom
df.info()

# Menampilkan statistik deskriptif dari dataset, dengan perhitungan persentil tambahan
df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

# Memeriksa jumlah nilai yang hilang (missing values) di setiap kolom dataset
df.isnull().sum()

"""## Data Preparation"""

# Mengisi nilai yang hilang pada kolom 'overview' dengan string kosong
df['overview'] = df['overview'].fillna('')

# Memastikan bahwa tidak ada lagi nilai yang hilang di kolom 'overview'
df['overview'].isnull().sum()

# Membuat objek TF-IDF Vectorizer dan menghapus stop words bahasa Inggris
tfidf = TfidfVectorizer(stop_words="english")

# Mengubah kolom 'overview' menjadi vektor numerik menggunakan TF-IDF
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Menampilkan dimensi matriks TF-IDF yang dihasilkan (baris x kolom)
tfidf_matrix.shape

# Memeriksa ukuran dari kolom judul film untuk dibandingkan dengan ukuran matriks TF-IDF
df['title'].shape

# Mengonversi matriks TF-IDF menjadi array (tidak selalu diperlukan, hanya untuk melihat isi matriksnya)
tfidf_matrix.toarray()

# Menghitung cosine similarity antara vektor TF-IDF untuk menemukan kesamaan antar film
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Menampilkan dimensi matriks cosine similarity yang dihasilkan
cosine_sim.shape

# Menampilkan kesamaan film pada indeks ke-1 dengan semua film lainnya
cosine_sim[1]

# Membuat Series di mana indeks adalah judul film dan nilai adalah indeks baris film
indices = pd.Series(df.index, index=df['title'])

# Menampilkan jumlah judul film yang duplikat
indices.index.value_counts()

# Menampilkan indeks dari film berjudul "Cinderella" (sebelum menghapus duplikat)
indices["Cinderella"]

# Menghapus judul film yang duplikat, hanya menyimpan yang terakhir
indices = indices[~indices.index.duplicated(keep='last')]

# Menampilkan indeks dari "Cinderella" setelah duplikat dihapus
indices["Cinderella"]

# Mengambil indeks film "Cinderella"
movie_index = indices["Cinderella"]

# Menampilkan skor kesamaan film "Cinderella" dengan semua film lainnya
cosine_sim[movie_index]

"""## Modeling and Result"""

# Membuat DataFrame yang berisi skor kesamaan antara "Cinderella" dan film lainnya
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

# Mengambil 10 film teratas yang mirip dengan "Cinderella", selain "Cinderella" itu sendiri
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

# Menampilkan judul film yang mirip dengan "Cinderella"
df['title'].iloc[movie_indices]

# Membuat fungsi untuk merekomendasikan film berdasarkan konten (overview)
def content_based_recommender(title, cosine_sim, dataframe):
    # Membuat indeks untuk setiap judul film
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    # Menghapus judul yang duplikat
    indices = indices[~indices.index.duplicated(keep='last')]
    # Mendapatkan indeks film yang sesuai dengan judul yang diberikan
    movie_index = indices[title]
    # Menghitung skor kesamaan film berdasarkan judul
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # Mengambil 10 film teratas yang mirip, kecuali film itu sendiri
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

# Mencoba fungsi rekomendasi untuk film "Minions"
content_based_recommender("Minions", cosine_sim, df)

# Mencoba fungsi rekomendasi untuk film "Family"
content_based_recommender("Family", cosine_sim, df)

"""## Conclusion

Kode ini berhasil mengimplementasikan sistem rekomendasi film berbasis konten menggunakan vektorisasi TF-IDF pada "overview" film. Dengan menghitung cosine similarity, sistem dapat menemukan 10 film teratas yang paling mirip berdasarkan konten teks "overview" film. Judul yang duplikat ditangani agar rekomendasi tetap unik, dan sistem ini dapat digunakan untuk merekomendasikan film apapun dalam dataset. Pendekatan ini efektif untuk rekomendasi berbasis konten, namun bisa ditingkatkan lebih lanjut dengan menambahkan fitur lain seperti genre atau rating pengguna.
"""