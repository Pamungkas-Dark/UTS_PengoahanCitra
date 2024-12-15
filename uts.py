# Import library yang diperlukan
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk histogram equalization
def histogram_equalization(image):
    # Hitung histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()  # Cumulative Distribution Function (CDF)
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalisasi CDF

    # Equalize histogram
    cdf_masked = np.ma.masked_equal(cdf, 0)  # Masking untuk mencegah pembagian nol
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')  # Isi kembali nilai masked dengan 0

    # Terapkan transformasi CDF ke citra asli
    image_equalized = cdf_final[image]
    return image_equalized

# Baca citra dengan kontras rendah (gunakan citra Anda sendiri)
image_path = 'low_contrast_image.jpg'  # Ganti dengan path citra Anda
# Gunakan mode='L' untuk citra grayscale
image = imageio.imread(image_path, mode='L')

# Terapkan histogram equalization
image_equalized = histogram_equalization(image)

# Tampilkan hasil
plt.figure(figsize=(12, 6))

# Citra asli
plt.subplot(1, 2, 1)
plt.title('Citra Asli (Kontras Rendah)')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Citra setelah histogram equalization
plt.subplot(1, 2, 2)
plt.title('Citra Setelah Histogram Equalization')
plt.imshow(image_equalized, cmap='gray')
plt.axis('off')

plt.show()
