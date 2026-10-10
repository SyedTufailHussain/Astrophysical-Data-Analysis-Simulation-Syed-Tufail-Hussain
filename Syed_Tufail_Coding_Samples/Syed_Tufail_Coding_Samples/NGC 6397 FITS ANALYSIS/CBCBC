import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from numpy.fft import fft2, fftshift

def load_fits_data(filename):
    """
    This function is opening the FITS file and converting the primary image to float.
    """
    with fits.open(filename) as hdul:
        data = hdul[0].data.astype(float)
    return data

def main():
    
    # 1) We are specifying our FITS file name here.
    fits_filename = "6397.fits"

    # 2) We are loading the FITS data from the file.
    image_data = load_fits_data(fits_filename)
    print("Data shape:", image_data.shape)

    # 3) We are estimating the background by using sigma-clipped statistics.
    mean_val, median_val, std_val = sigma_clipped_stats(image_data, sigma=3.0)
    print(f"Background median = {median_val:.3f}, std = {std_val:.3f}")

    # 4) We are subtracting the median background so that real signals stand out.
    data_sub = image_data - median_val

    # 5) We are masking bright sources above 5σ to remove big stars or galaxies.
    bright_threshold = median_val + 5 * std_val
    data_masked = data_sub.copy()
    data_masked[data_masked > bright_threshold] = np.nan

    # 6) We are computing a 2D power spectrum to see how brightness varies at different scales.
    data_no_nan = np.nan_to_num(data_masked, nan=0.0)
    fft_result = fft2(data_no_nan)
    fft_shifted = fftshift(fft_result)
    power_spectrum = np.abs(fft_shifted)**2

    # 7) We are putting all images into one figure with subplots.

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Add a main title to the entire figure
    fig.suptitle("6397 FITS Analysis", fontsize=20, fontweight = 'bold')

    # (A) Original Image
    axes[0, 0].imshow(
        image_data,
        origin='lower',
        cmap='inferno',
        vmin=0,
        vmax=np.percentile(image_data, 99)
    )
    axes[0, 0].set_title("Original Image")

    # (B) Background-Subtracted
    axes[0, 1].imshow(
        data_sub,
        origin='lower',
        cmap='inferno',
        vmin=-std_val,
        vmax=np.percentile(data_sub[np.isfinite(data_sub)], 99)
    )
    axes[0, 1].set_title("Background-Subtracted")

    # (C) Masked (5σ)
    axes[1, 0].imshow(
        data_masked,
        origin='lower',
        cmap='inferno',
        vmin=-std_val,
        vmax=np.percentile(data_sub[np.isfinite(data_sub)], 99)
    )
    axes[1, 0].set_title("Masked (5σ)")

    # (D) 2D Power Spectrum (log scale)
    axes[1, 1].imshow(
        np.log10(power_spectrum + 1e-8),
        origin='lower',
        cmap='magma'
    )
    axes[1, 1].set_title("2D Power Spectrum (log)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
