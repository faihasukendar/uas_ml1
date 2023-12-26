# Laporan Proyek Machine Learning
### Nama : Faiha Atsaa S
### Nim : 211351053
### Kelas : Malam B

## Domain Proyek

Memudahkan kita untuk mengestimasi harga rumah di boston berdasarkan 8 attribut tanpa harus benar benar mengunjungi lokasinya


## Business Understanding

Dengan adanya aplikasi ini, sangat memudahkan kita untuk melakukan estimasi harga rumah di boston berdasarkan attribut yang di tawarkan. Sehingga kita bisa menyiapkan dana yang sesuai dan hanya perlu sekali saja pergi ke boston keetika pindah tidak untuk bulak balik cek harga pasar.

Bagian laporan ini mencakup:

### Problem Statements

Seringkali ketika ingin membeli rumah di luar negri, alih alih sekali deal kita harus malah bulak balik mencari rumah yang cocok sehingga tabungan kita untuk membeli rumah malah habis terpakai untuk ongkos

### Goals

Dengan hadirnya aplikasi ini memudahkan kita mendapat estimasi harga rumah yang cocok dengan kita dan dapat menghemat biaya akomodasi kita sehingga hanya perlu sekali berangkat ke boston untuk proses transaksi dan pindahan.


   ### Solution statements
    - Karena model aplikasi ini berbentuk sebuah estimasi maka digunakanlah algoritma linear regresi untuk menemukan harga yang tepat

## Data Understanding
Data ini berdasarkan dataset dari kaggle tentang harga rumah di boston.<br> 

[Boston House Data](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data).


### Variabel-variabel pada Boston House Price Dataset adalah sebagai berikut:
1) CRIM: Rata rata tingkat kriminal dalam setahun terakhir
2) ZN: Tingkat kepadatan penduduk per 25000 sq.ft
3) INDUS: Tingkat kepadatan area industri
4) CHAS: Apakah dekat sungai atau tidak
5) NOX: nitric oxides concentration (parts per 10 juta) [parts/10M]
6) RM: Rata rata ruangan di daerah
7) AGE: Usia bangunan
8) DIS: Total jarak dari pusat pekerjaan
9) RAD: Tingkat akses terhadap jalan tol
10) TAX: rata rata pajak  per $10,000 [$/10k]
11) PTRATIO: tenaga pengajar di daearh
12) B: Hasil dari B=1000(Bk - 0.63)^2 dimana Bk adalah tingkat populasi kulit hitam
13) LSTAT: persentase status penduduk kelas bawah

## Data Preparation
Seteleh menentukan dataset yang akan dibuatkan model Machine Learningnya selanjutnya kita ketikan library python yang ingin di gunakan
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```
Setelah itu kita buka file datasetnya
```bash
df = pd.read_csv('the-boston-houseprice-data/boston.csv')
```
Kita cek apakah datanya sudah terbaca atau belum
```bash
df.head()
```
Kita bisa lihat informasi table nya dengan cara
```bash
df.info()
```
Maka akan muncul
```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 14 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   CRIM     506 non-null    float64
 1   ZN       506 non-null    float64
 2   INDUS    506 non-null    float64
 3   CHAS     506 non-null    int64  
 4   NOX      506 non-null    float64
 5   RM       506 non-null    float64
 6   AGE      506 non-null    float64
 7   DIS      506 non-null    float64
 8   RAD      506 non-null    int64  
 9   TAX      506 non-null    float64
 10  PTRATIO  506 non-null    float64
 11  B        506 non-null    float64
 12  LSTAT    506 non-null    float64
 13  MEDV     506 non-null    float64
dtypes: float64(12), int64(2)
memory usage: 55.5 KB
```
Terlihat lah setiap kolom dan tipe datanya<br>
Selanjutnya kita bisa visualisasikan data tersebut dalam bentuk heatmap
```bash
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```
![alt text](https://github.com/faihasukendar/pembelajaranmesin/blob/main/heatmap.png)
Bisa dilihat bahwa data dari setiap kolomnya tervisualisai

```bash
sns.set_style('whitegrid')
sns.displot(df['MEDV'], kde=True)
plt.title('Distribution of median value of owner-occupied homes')
```
![download (1)](https://github.com/faihasukendar/pembelajaranmesin/assets/149061885/8d6ab171-7452-49a2-be16-9f475cf13ec8)

```bash
sns.lmplot(x='CRIM', y='MEDV', data=df, aspect=2)
plt.xlabel('capita crime rate by town')
plt.ylabel("Median value of owner-occupied homes in 1000's")
plt.title('houses pricing vs crime rate')
```
![download](https://github.com/faihasukendar/pembelajaranmesin/assets/149061885/c4272984-3242-49e7-bccd-1e3055a0c0b5)


Sebenarnya masih banyak cara untuk memvisualisasikan data pada proses EDA ini tapi mari kita lanjutkan ke proses modeling.

## Modeling
Pertama kita tentukan dulu fitur untuk X dan label untuk Y
```bash
x = df.drop (columns='MEDV', axis=1)
y = df['MEDV']
x.shape, y.shape
```
Jika sudah ditentukan maka bisa kita lanjutkan dengan melakukan data train dan test
```bash
x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=70)
```
Jika sudah maka kita masukan algortima linear regresi dengan nilai X dan Y
```bash
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```
Sampai tahap ini proses modeling sudah selesai dan bisa dilakukan pengetesan dengan cara
```bash
input_data = np.array([[0.00632,18.0,2.31,0,0.538,6.575,65.2,4.0900,1,296.0,15.3,396.90,4.98]])

prediction = lr.predict(input_data)
print('Estimasi Harga Rumah :', prediction)
```
```bash
Estimasi Harga Rumah : [30.28681756]
```
Maka akan muncul harga estimasinya


## Evaluation
Data ini di evaluasi melalui nilai akurasinya

![Screenshot (52)](https://github.com/faihasukendar/pembelajaranmesin/assets/149061885/237b2c06-b842-471d-be96-43abd457238f)



## Deployment
[The Boston House Price](https://faihautsml.streamlit.app/)<br>
![alt text](https://github.com/faihasukendar/pembelajaranmesin/blob/main/tampilan.png)
