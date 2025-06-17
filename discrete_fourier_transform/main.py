import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

image_path = 'images/cameraman.tif'
small_image_path = 'images/cameraman_small.tif'

def load_image(path, grayscale=True):
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    return np.array(img)

def save_image(image_array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, image_array, cmap='gray', vmin=0, vmax=255)

def save_spectrum_image(dft, path):
    magnitude_spectrum = np.abs(dft)
    log_spectrum = np.log1p(magnitude_spectrum) 
    log_spectrum = 255 * (log_spectrum / np.max(log_spectrum))
    log_spectrum = log_spectrum.astype(np.uint8)
    save_image(log_spectrum, path)

def reconstruct_from_real(dft):
    real_dft = np.real(dft)
    complex_real = real_dft + 0j
    # Use only the real part of the IFFT result, clip to valid range
    return np.clip(np.fft.ifft2(complex_real).real, 0, 255).astype(np.uint8)

def reconstruct_from_imag(dft):
    imag_dft = np.imag(dft)
    complex_imag = 1j * imag_dft
    # Use only the real part of the IFFT result, clip to valid range
    return np.clip(np.fft.ifft2(complex_imag).real, 0, 255).astype(np.uint8)

def my_idft(dft):
    N, M = dft.shape
    output = np.zeros((N, M), dtype=complex)

    for x in range(N):
        for y in range(M):
            sum_val = 0.0
            for u in range(N):
                for v in range(M):
                    angle = 2j * np.pi * ((u * x) / N + (v * y) / M)
                    sum_val += dft[u, v] * np.exp(angle)
            output[x, y] = sum_val / (N * M)

    return np.clip(output.real, 0, 255).astype(np.uint8)

def my_dft(image):
    N, M = image.shape
    output = np.zeros((N, M), dtype=complex)

    for u in range(N):
        for v in range(M):
            sum_val = 0.0
            for x in range(N):
                for y in range(M):
                    angle = -2j * np.pi * ((u * x) / N + (v * y) / M)
                    sum_val += image[x, y] * np.exp(angle)
            output[u, v] = sum_val

    return output

def remove_dc_component(image):
    dft = np.fft.fft2(image)
    dft[0, 0] = 0  # Set DC term to zero
    idft = np.fft.ifft2(dft).real
    return np.clip(idft, 0, 255).astype(np.uint8)

def subtract_average_intensity(image):
    mean_val = np.mean(image)
    adjusted = image - mean_val
    adjusted = np.clip(adjusted, 0, 255)  # Recenter for display
    return adjusted.astype(np.uint8)
    
def reconstruct_from_fftshift(image):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    recon = np.fft.ifft2(dft_shifted).real
    return np.clip(recon, 0, 255).astype(np.uint8)

def spatial_domain_shift(image):
    N, M = image.shape
    x = np.arange(N).reshape(-1, 1)
    y = np.arange(M).reshape(1, -1)
    sign_matrix = (-1) ** (x + y)
    g = image * sign_matrix
    return np.clip(g, 0, 255).astype(np.uint8)

def spatial_fftshift(image):
    shifted = np.fft.fftshift(image)
    return np.clip(shifted, 0, 255).astype(np.uint8)

def reconstruct_from_amplitude(dft):
    amplitude = np.abs(dft)
    new_spectrum = amplitude.astype(complex)
    reconstructed = np.fft.ifft2(new_spectrum)
    return reconstructed.real

def reconstruct_from_phase(dft):
    phase = np.angle(dft)
    new_spectrum = np.exp(1j * phase)
    reconstructed = np.fft.ifft2(new_spectrum).real

    reconstructed -= reconstructed.min()
    reconstructed /= reconstructed.max()
    reconstructed *= 255

    return reconstructed.astype(np.uint8)

def task_one():
    image = load_image(image_path)
    dft = np.fft.fft2(image)
    real_reconstructed = reconstruct_from_real(dft)
    imag_reconstructed = reconstruct_from_imag(dft)

    save_image(real_reconstructed, 'results/real_reconstructed.png')
    save_image(imag_reconstructed, 'results/imag_reconstructed.png')

def task_two():

    # a) np.fft -> my_idft
    image = load_image(small_image_path)
    dft = np.fft.fft2(image)
    my_reconstructed = my_idft(dft)
    save_image(my_reconstructed, 'results/my_idft_reconstructed.png')

    # b) my_dft -> np.ifft
    dft_my = my_dft(image)
    np_reconstructed = np.fft.ifft2(dft_my)
    np_reconstructed = np.clip(np.real(np_reconstructed), 0, 255).astype(np.uint8)
    save_image(np_reconstructed, 'results/np_ifft_reconstructed.png')

    # c) my_dft -> my_idft
    dft_my = my_dft(image)
    my_reconstructed = my_idft(dft_my)
    save_image(my_reconstructed, 'results/my_idft_reconstructed.png')

def task_three():
    image = load_image(image_path)

    # remove dc term in dft and reconstruct
    dc_removed = remove_dc_component(image)
    save_image(dc_removed, 'results/dc_removed.png')

    # subtract mean intensity
    mean_subtracted = subtract_average_intensity(image)
    save_image(mean_subtracted, 'results/mean_subtracted.png')

def task_four():
    image = load_image(image_path)

    # reconstruct from fftshifted DFT
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    save_spectrum_image(dft, 'results/spectrum.png')
    save_spectrum_image(dft_shifted, 'results/fftshifted_spectrum.png')

    recon_from_shifted_spectrum = reconstruct_from_fftshift(image)
    save_image(recon_from_shifted_spectrum, 'results/fftshift_reconstructed.png')

    # applying fftshift in the frequency domain is the to multiply the image by (-1)^(x+y) in the spatial domain
    shifted_image = spatial_domain_shift(image)
    save_image(shifted_image, 'results/spatial_domain_shifted.png')

    # apply fftshift directly to image in the spatial domain
    fftshifted_image = spatial_fftshift(image)
    save_image(fftshifted_image, 'results/spatial_fftshifted.png')
    dft = np.fft.fft2(fftshifted_image)
    save_spectrum_image(dft, 'results/spatial_fftshifted_spectrum.png')

def task_five():
    image = load_image(image_path)
    dft = np.fft.fft2(image)
    amp_recon = reconstruct_from_amplitude(dft)
    phase_recon = reconstruct_from_phase(dft)
    save_image(amp_recon, 'results/reconstructed_amplitude.png')
    save_image(phase_recon, 'results/reconstructed_phase.png')

def main():
    task_one()
    task_two()
    task_three()
    task_four()
    task_five()

if __name__ == '__main__':
    main()
