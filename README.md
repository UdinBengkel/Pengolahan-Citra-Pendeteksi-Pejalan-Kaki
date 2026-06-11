# Tugas Chapter 15 — Deteksi Pejalan Kaki menggunakan OpenCV-Python

| | |
|---|---|
| **Nama** | *Syafarudiansya* |
| **NIM** | *312410381* |
| **Kelas** | I241A |
| **Mata Kuliah** | Pengolahan Citra |
| **Topik** | Deteksi Pejalan Kaki dengan HOG + Linear SVM |

---

## Daftar Isi
1. [Pendahuluan](#pendahuluan)
2. [Dasar Teori](#dasar-teori)
3. [Persyaratan](#persyaratan)
4. [Contoh 1 — Deteksi dari Gambar](#contoh-1--deteksi-dari-gambar)
5. [Contoh 2 — Deteksi dari Video](#contoh-2--deteksi-dari-video)
6. [Kesimpulan](#kesimpulan)

---

## Pendahuluan

OpenCV (*Open Source Computer Vision Library*) adalah pustaka sumber terbuka yang ditujukan untuk visi komputer waktu nyata. Dikembangkan oleh Intel, pustaka ini bersifat lintas platform dan mendukung berbagai bahasa pemrograman seperti Python, C++, dan Java.

Dalam tugas ini, dibangun program **Deteksi Pejalan Kaki** untuk gambar statis dan aliran video menggunakan metode bawaan OpenCV, yaitu **HOG + Linear SVM** yang telah dilatih sebelumnya.

Deteksi pejalan kaki merupakan bidang penelitian penting karena berperan dalam sistem keselamatan kendaraan otonom — membantu mobil mengenali keberadaan manusia di sekitarnya secara real-time.

---

## Dasar Teori

### HOG (Histogram of Oriented Gradients)

HOG adalah algoritma ekstraksi fitur yang bekerja dengan cara:

1. **Memeriksa piksel sekitar** — setiap piksel dibandingkan dengan piksel di sekitarnya untuk menghitung seberapa gelap piksel saat ini.
2. **Menggambar gradien** — algoritma menggambar panah (*gradient*) yang menunjukkan arah dari terang ke gelap pada setiap piksel.
3. **Analisis histogram** — gradien dikelompokkan ke dalam histogram berdasarkan orientasinya untuk menghasilkan vektor fitur.

Vektor fitur HOG kemudian diumpankan ke **Linear SVM** (Support Vector Machine) yang telah dilatih menggunakan dataset manusia, sehingga model dapat mengenali pola tubuh manusia dalam gambar baru.

> Referensi: Navneet Dalal & Bill Triggs, *"Histograms of Oriented Gradients for Human Detection"*, CVPR 2005.

---

## Persyaratan

```bash
pip install opencv-python
pip install imutils
```

| Library | Fungsi |
|---|---|
| `opencv-python` | Pemrosesan gambar & video, HOG descriptor |
| `imutils` | Utilitas resize gambar dengan mudah |

---

## Contoh 1 — Deteksi dari Gambar

### Kode Program

```python
import cv2
import imutils

# Inisialisasi HOG People Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca gambar
image = cv2.imread('dude.jpg')

# Resize gambar (max lebar 400px)
image = imutils.resize(image, width=min(400, image.shape[1]))

# Deteksi pejalan kaki
(regions, _) = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(4, 4),
    scale=1.05
)

# Gambar bounding box
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

print(f"Jumlah pejalan kaki terdeteksi: {len(regions)}")

cv2.imwrite('output_deteksi.png', image)
cv2.imshow("Deteksi Pejalan Kaki", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Penjelasan Parameter `detectMultiScale`

| Parameter | Nilai | Keterangan |
|---|---|---|
| `winStride` | `(4, 4)` | Langkah pergeseran sliding window (px). Lebih kecil = lebih akurat, lebih lambat |
| `padding` | `(4, 4)` | Padding di sekitar window sebelum komputasi HOG |
| `scale` | `1.05` | Faktor skala image pyramid. Lebih kecil = lebih banyak skala yang dicek |

### Hasil

> 📸 *Letakkan screenshot hasil output di sini*
> Contoh: `![Hasil Contoh 1](img/output_contoh1.png)`

---

## Contoh 2 — Deteksi dari Video

### Kode Program

```python
import cv2
import imutils

# Inisialisasi HOG People Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('aura.mp4')

# Simpan video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(min(400, cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) *
             (width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))

        (regions, _) = hog.detectMultiScale(
            image,
            winStride=(4, 4),
            padding=(4, 4),
            scale=1.05
        )

        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Tampilkan jumlah deteksi di frame
        cv2.putText(image, f"Pejalan Kaki: {len(regions)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        out.write(image)
        cv2.imshow("Deteksi Pejalan Kaki - Video", image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

### Perbedaan dengan Contoh 1

| Aspek | Contoh 1 (Gambar) | Contoh 2 (Video) |
|---|---|---|
| Input | File gambar statis `.jpg/.png` | File video `.mp4` frame by frame |
| Loop | Tidak ada | `while cap.isOpened()` membaca tiap frame |
| Output | Gambar hasil `.png` | Video hasil `.mp4` via `VideoWriter` |
| Info real-time | Tidak ada | Counter pejalan kaki di tiap frame |
| Keluar | `cv2.waitKey(0)` | Tekan `q` untuk stop |

### Dokumentasi Hasil (Screenshot Frame)

<img width="400" height="224" alt="Image" src="https://github.com/user-attachments/assets/bd6d1768-b226-482a-a86a-867c49c4b2e1" />
---

## Kesimpulan

1. OpenCV menyediakan metode bawaan deteksi pejalan kaki berbasis **HOG + Linear SVM** yang efektif tanpa perlu melatih model dari awal.
2. Algoritma HOG bekerja dengan mengekstraksi arah gradien piksel untuk membentuk vektor fitur yang merepresentasikan bentuk tubuh manusia.
3. Program **Contoh 1** berhasil mendeteksi pejalan kaki dalam gambar statis dengan menggambar bounding box merah di sekitar area yang terdeteksi.
4. Program **Contoh 2** memperluas implementasi ke video — memproses tiap frame secara berurutan dan menyimpan hasilnya sebagai video output.
5. Parameter `winStride`, `padding`, dan `scale` pada `detectMultiScale` sangat berpengaruh terhadap akurasi dan kecepatan deteksi.
