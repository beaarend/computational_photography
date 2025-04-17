import cv2
import rawpy
import numpy as np
import argparse

dng_path = 'input_images/converted_raw_image.dng'

def load_dng_file(dng_path):
    raw = rawpy.imread(dng_path)
    raw_array = raw.raw_image_visible

    return raw_array

def bilinear_demosaic(bayer_data):
    h, w = bayer_data.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint16)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if (i % 2 == 0) and (j % 2 == 0):  # Red pixel
                rgb_image[i, j, 0] = bayer_data[i, j]
                rgb_image[i, j, 1] = np.mean([bayer_data[i-1, j], bayer_data[i+1, j], bayer_data[i, j-1], bayer_data[i, j+1]])
                rgb_image[i, j, 2] = np.mean([bayer_data[i-1, j-1], bayer_data[i-1, j+1], bayer_data[i+1, j-1], bayer_data[i+1, j+1]])
            elif (i % 2 == 1) and (j % 2 == 1):  # Blue pixel
                rgb_image[i, j, 2] = bayer_data[i, j]
                rgb_image[i, j, 1] = np.mean([bayer_data[i-1, j], bayer_data[i+1, j], bayer_data[i, j-1], bayer_data[i, j+1]])
                rgb_image[i, j, 0] = np.mean([bayer_data[i-1, j-1], bayer_data[i-1, j+1], bayer_data[i+1, j-1], bayer_data[i+1, j+1]])
            else:  # Green pixel
                rgb_image[i, j, 1] = bayer_data[i, j]
                rgb_image[i, j, 0] = np.mean([bayer_data[i, j-1], bayer_data[i, j+1]]) if i % 2 == 0 else np.mean([bayer_data[i-1, j], bayer_data[i+1, j]])
                rgb_image[i, j, 2] = np.mean([bayer_data[i-1, j], bayer_data[i+1, j]]) if i % 2 == 0 else np.mean([bayer_data[i, j-1], bayer_data[i, j+1]])
    return rgb_image

def white_balance(image, type):

    if type == "gray_world":
        r_wb = np.mean(image[:,:,0])
        g_wb = np.mean(image[:,:,1])
        b_wb = np.mean(image[:,:,2])
        alpha = g_wb/r_wb
        beta = g_wb/b_wb

        image[:, :, 0] = np.clip(image[:, :, 0] * alpha, 0, 255).astype(np.uint8)
        image[:, :, 2] = np.clip(image[:, :, 2] * beta, 0, 255).astype(np.uint8)

    elif type == "white_patch":
        max_r = np.max(image[:, :, 0])
        max_g = np.max(image[:, :, 1])
        max_b = np.max(image[:, :, 2])
        
        image[:, :, 0] = image[:, :, 0] * (255 / max_r)
        image[:, :, 1] = image[:, :, 1] * (255 / max_g)
        image[:, :, 2] = image[:, :, 2] * (255 / max_b)

def gamma_correction(image, gamma):

    # Gamma correction
    image = image / 255.0
    image = np.power(image, gamma)
    image = (image * 255).astype(np.uint8)

    return image

def main():
    
    # load dng files
    print("Loading DNG file...")
    bayer_data = load_dng_file(dng_path)
    bayer_data_8bit = (bayer_data / bayer_data.max() * 255).astype(np.uint8)
    cv2.imwrite("result_images/dng_image.png", bayer_data_8bit)

    # demosaicking
    print("Demosaicking...")
    interpolated_image = bilinear_demosaic(bayer_data)
    interpolated_image = rggb_demosaic(bayer_data)
    interpolated_image_8bit = (interpolated_image / interpolated_image.max() * 255).astype(np.uint8)
    interpolated_image_bgr = cv2.cvtColor(interpolated_image_8bit, cv2.COLOR_RGB2BGR)
    cv2.imwrite("result_images/grbg_image.png", interpolated_image_bgr)

    # white balance
    print("White balancing...")
    image = cv2.imread("result_images/grbg_image.png")
    white_balance(image, type="gray_world")
    cv2.imwrite("result_images/gw_white_balanced_image.png", image)
    white_balance(image, type="white_patch")
    cv2.imwrite("result_images/wp_white_balanced_image.png", image)

    # gamma correction
    print("Gamma correction...")
    image = cv2.imread("result_images/gw_white_balanced_image.png")
    for gamma in [0.5, 0.75, 0.85, 1.25]:
        corrected_image = gamma_correction(image, gamma)
        cv2.imwrite(f"result_images/gamma_corrected_image_{gamma}.png", corrected_image)


if __name__ == "__main__":
    main()