
# Laporan Proyek Machine Learning - Prediksi Penjualan Kopi

## Domain Proyek
Pada proyek ini, fokus utama adalah prediksi penjualan kopi untuk membantu meningkatkan pengelolaan stok inventori secara lebih efisien. Dengan menganalisis tren penjualan dari data historis, kita dapat memproyeksikan jumlah permintaan kopi dalam periode tertentu.

### Mengapa Masalah Ini Penting?
Mengoptimalkan persediaan sangat penting untuk menghindari kekurangan atau kelebihan stok, yang dapat berdampak pada biaya operasional atau kehilangan penjualan. Dengan adanya prediksi yang lebih akurat, pengelolaan persediaan dapat lebih optimal, sehingga mengurangi biaya sambil meningkatkan ketersediaan produk.

## Business Understanding
Dalam proyek ini, pernyataan masalah dan tujuan dari analisis ini adalah:

### Problem Statements:
1. Bagaimana memprediksi permintaan kopi harian?
2. Bagaimana meningkatkan penjualan atau mengoptimalkan pengelolaan persediaan berdasarkan prediksi tersebut?

### Goals:
1. Membuat model prediksi penjualan yang dapat diandalkan dengan menggunakan data historis penjualan kopi.
2. Meningkatkan pengelolaan stok inventori dengan mendasarkan keputusan pada hasil prediksi penjualan.

### Solution Statements:
Untuk menyelesaikan masalah ini, beberapa solusi yang diajukan adalah:
- Menggunakan model ARIMA untuk prediksi time series.
- Menggunakan model Auto-SARIMAX sebagai alternatif dan membandingkan kinerjanya dengan ARIMA.
- Melakukan evaluasi model menggunakan Mean Absolute Error (MAE) sebagai metrik evaluasi utama.

## Data Understanding

Dataset yang digunakan adalah data penjualan kopi yang diperoleh dari Kaggle: [Coffee Sales Dataset](https://www.kaggle.com/datasets/ihelon/coffee-sales) dengan total **1,917** baris dan **6** kolom. Data ini mencakup waktu transaksi, jenis pembayaran, jumlah uang yang dibelanjakan, serta nama kopi yang dibeli.

Berikut adalah rincian tiap kolom:

1. **date**: Tanggal transaksi (tipe `object`).
2. **datetime**: Waktu lengkap transaksi (tipe `object`).
3. **cash_type**: Jenis pembayaran, misalnya menggunakan kartu (tipe `object`).
4. **card**: Anonimisasi nomor kartu yang digunakan untuk transaksi (tipe `object`). Terdapat **89 nilai kosong** di kolom ini.
5. **money**: Jumlah uang yang dibelanjakan (tipe `float64`).
6. **coffee_name**: Nama kopi yang dibeli (tipe `object`).

### Statistik Deskriptif untuk Kolom `money`

- **Rata-rata (mean)**: 31.56
- **Standar deviasi (std)**: 5.26
- **Nilai minimum (min)**: 18.12
- **Kuartil 25%**: 27.92
- **Median (50%)**: 32.82
- **Kuartil 75%**: 35.76
- **Nilai maksimum (max)**: 40.00

### Nilai Hilang
Kolom **card** memiliki **89 nilai kosong**, sementara kolom lainnya tidak memiliki nilai yang hilang.

### Duplikasi Data
Tidak ada data duplikat dalam dataset ini, seperti ditunjukkan oleh hasil `df.duplicated().sum()`, yang menghasilkan 0.

### Analisis Awal
- Kolom **money** menunjukkan nilai minimum sebesar 18.12 dan maksimum 40.00, dengan distribusi yang cukup merata berdasarkan kuartilnya.
- Kolom **card** memiliki beberapa nilai kosong yang perlu diatasi dalam proses **Data Preparation**.

Tahapan selanjutnya adalah membersihkan nilai kosong dan memastikan bahwa data siap untuk dianalisis lebih lanjut.

## Data Preparation

Pada tahap ini, dilakukan beberapa persiapan data agar siap untuk digunakan dalam pemodelan. Langkah-langkah yang dilakukan adalah sebagai berikut:

### 1. Mengonversi Kolom `date` Menjadi Tipe Datetime
Agar bisa melakukan analisis waktu, kolom **date** diubah dari tipe `object` ke tipe `datetime`.

```python
df["date"] = pd.to_datetime(df["date"])
```

### 2. Membuat Rangkaian Tanggal yang Lengkap
Pada tahap ini, dilakukan beberapa persiapan data agar siap untuk digunakan dalam pemodelan. Untuk memastikan bahwa data memiliki rangkaian tanggal yang lengkap, dibuat rentang waktu dari tanggal awal hingga tanggal akhir dalam dataset. Ini dilakukan dengan pd.date_range(), dan kemudian digabungkan dengan data penjualan berdasarkan tanggal.

```python
date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
complete_dates = pd.DataFrame(date_range, columns=["date"])
df_by_date = df.groupby("date").agg({"money": ["count"]}).reset_index()
df_by_date.columns = ["date", "cups"]
df_complete = pd.merge(complete_dates, df_by_date, on="date", how="left")
df_complete.fillna(0, inplace=True)
```

- complete_dates: Tabel dengan semua tanggal dari periode data, termasuk tanggal yang tidak ada di dataset asli.
- df_by_date: Tabel yang mengelompokkan jumlah transaksi per hari, disimpan dalam kolom cups.
- df_complete: Penggabungan kedua tabel di atas, dengan pengisian nilai kosong dengan 0 untuk hari tanpa transaksi.

### 3. Menghitung Korelasi antar Variabel
Menggunakan heatmap untuk menampilkan korelasi antar variabel yang tersedia dalam dataset.

```pyhton
corr = df_complete.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)
```

### 4. Split Train dan Test
Data dibagi menjadi set pelatihan dan pengujian, di mana 7 hari terakhir digunakan sebagai data uji.

```python
test_size = 7
train_size = df_complete.shape[0] - test_size

df_train = df_complete.iloc[:train_size]
df_test = df_complete.iloc[train_size:]
```

- train_size: Ukuran data pelatihan yang merupakan keseluruhan data dikurangi 7 hari untuk pengujian.
- df_train: Data pelatihan.
- df_test: Data pengujian.

### 5. Visualisasi Data Train dan Test
Data pelatihan dan pengujian divisualisasikan untuk melihat tren.

```python
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="cups", x="date", label="Train")
sns.lineplot(data=df_test, y="cups", x="date", label="Test")
plt.grid()
plt.ylim(0)
```

Langkah ini membantu memahami bagaimana data dibagi dan melihat bagaimana distribusi jumlah transaksi (cups) berdasarkan waktu.

## Model Development.
Pada tahap ini, saya menggunakan dua jenis model untuk memprediksi jumlah cangkir kopi yang terjual: ARIMA dan Auto-SARIMAX. Saya akan menjelaskan proses pengembangan model, parameter yang digunakan, serta bagaimana masing-masing model bekerja.

### ARIMA (AutoRegressive Integrated Moving Average)
Model ARIMA adalah salah satu metode yang populer dalam pemodelan deret waktu. Pada model ini, saya menggunakan tiga parameter utama:

```pyhton
p, d, q = 5, 0, 5
model = ARIMA(df_train['cups'], order=(p, d, q))
model_fit = model.fit()

test_predictions = model_fit.forecast(steps=len(df_test)).values
df_test["arima_pred"] = test_predictions
```

p (AutoRegressive Order): Nilai ini menunjukkan berapa banyak lag atau nilai sebelumnya dari data yang digunakan untuk memprediksi nilai saat ini. Dalam kasus ini, saya memilih p = 5, yang berarti model mempertimbangkan lima data sebelumnya.
d (Differencing): Parameter ini digunakan untuk mengubah data agar menjadi stasioner, atau menghilangkan tren dari data. Saya memilih d = 0 karena data sudah dianggap stasioner.
q (Moving Average Order): Nilai ini menentukan berapa banyak residual atau kesalahan dari model sebelumnya yang digunakan dalam prediksi. Saya memilih q = 5.
Saya kemudian memisahkan data menjadi train dan test, melatih model ARIMA pada data train, dan memprediksi jumlah cangkir kopi terjual untuk data test. Nilai prediksi dibandingkan dengan data aktual untuk menghitung kesalahan dengan Mean Absolute Error (MAE).

### Auto-SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)
Auto-SARIMAX adalah pengembangan dari ARIMA yang secara otomatis memilih parameter terbaik untuk model deret waktu dan juga mempertimbangkan faktor musiman. Parameter utama yang digunakan dalam Auto-SARIMAX adalah:

```python
model = auto_arima(
    df_train['cups'],
    seasonal=True,
    m=7,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
)

test_predictions = model.predict(n_periods=len(df_test)).values
df_test["auto_sarimax_pred"] = test_predictions
```

seasonal=True: Ini menandakan bahwa model mempertimbangkan pola musiman dalam data.
m=7: Parameter ini menunjukkan periode musiman. Karena data mewakili penjualan harian, saya menggunakan m = 7, yang berarti model akan menangkap pola mingguan.
trace=True: Ini digunakan untuk menampilkan proses iterasi pemilihan parameter secara otomatis.
Auto-SARIMAX secara otomatis mencari parameter yang paling optimal untuk model, dan sama seperti ARIMA, model ini juga digunakan untuk memprediksi nilai pada data test, diikuti dengan evaluasi menggunakan MAE.

## Evaluation
Pada tahap evaluasi, tujuan utama adalah menilai performa model dalam memprediksi jumlah cangkir kopi terjual, serta melihat sejauh mana hasil prediksi tersebut dapat membantu dalam menyelesaikan masalah terkait manajemen persediaan kopi.

Keterkaitan dengan Problem Statement
Masalah yang ingin diselesaikan adalah memprediksi penjualan kopi harian untuk membantu manajemen persediaan, agar dapat dilakukan pengelolaan stok yang lebih efisien, menghindari kekurangan atau kelebihan stok. Untuk mencapai tujuan ini, model prediksi harus dapat memberikan estimasi yang akurat terhadap penjualan harian di masa mendatang.

Hasil Evaluasi Model
Dua model yang digunakan, ARIMA dan Auto-SARIMAX, dievaluasi menggunakan metrik Mean Absolute Error (MAE), yang mengukur rata-rata besarnya kesalahan antara prediksi dan data aktual. Hasil MAE untuk kedua model adalah sebagai berikut:

ARIMA MAE = 6.35
Auto-SARIMAX MAE = 4.39
Dari hasil di atas, model Auto-SARIMAX memiliki kesalahan prediksi yang lebih kecil dibandingkan dengan model ARIMA, menunjukkan bahwa Auto-SARIMAX lebih baik dalam menangkap pola musiman penjualan kopi. Hal ini masuk akal karena Auto-SARIMAX secara eksplisit mempertimbangkan pola musiman mingguan yang terdapat dalam data penjualan.

Solusi dan Implikasi
Dengan menggunakan model Auto-SARIMAX, prediksi penjualan harian yang lebih akurat dapat membantu manajemen toko kopi dalam merencanakan persediaan. Model ini bisa digunakan untuk memperkirakan kebutuhan stok kopi setiap minggunya, sehingga dapat mencegah kekurangan stok pada hari-hari sibuk dan mengurangi biaya penyimpanan dari stok berlebih. Implementasi prediksi penjualan ini akan memberikan dampak positif pada efisiensi operasional dan peningkatan kepuasan pelanggan.

Berdasarkan hasil MAE dan visualisasi, jika Auto SARIMAX menghasilkan MAE yang lebih rendah, ini menandakan bahwa model tersebut lebih baik dalam memprediksi data pengujian. Sebaliknya, jika ARIMA memberikan nilai MAE yang lebih rendah, maka ARIMA dapat dianggap lebih tepat.
Selain itu, pastikan untuk mempertimbangkan simplicity (kesederhanaan) dan computational cost (biaya komputasi) dalam memilih model. Jika Auto SARIMAX memberikan hasil yang sedikit lebih baik tetapi dengan waktu pelatihan yang jauh lebih lama, ARIMA mungkin tetap menjadi pilihan yang lebih efisien tergantung pada konteks dan kebutuhan.
Dengan menambahkan perbandingan ini, akan bisa menentukan model mana yang paling cocok untuk prediksi penjualan berdasarkan MAE dan visualisasi hasil prediksi.

Dengan demikian, hasil evaluasi model ini memberikan solusi yang relevan dengan masalah yang dihadapi, yakni perencanaan stok yang lebih akurat berdasarkan prediksi penjualan.