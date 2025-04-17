import cv2
import numpy as np
import matplotlib.pyplot as plt

exposure_times = [0.0333, 0.1000, 0.3333, 0.6520, 1.0000, 4.0000]
image_paths = [
    'images/office_1.jpg',
    'images/office_2.jpg',
    'images/office_3.jpg',
    'images/office_4.jpg',
    'images/office_5.jpg',
    'images/office_6.jpg'
]

def load_curve(path):
    curve = np.loadtxt(path)
    response_curve = np.exp(curve)

    return response_curve

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Image at {path} could not be loaded.")
        images.append(img)

    return images

def convert_images(images):
    conv_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    float_images = [img.astype(np.uint8) for img in conv_images]
    return float_images

def create_hdr_image(ldr_images, exposure_times, response_curve):

    height, width, _ = ldr_images[0].shape
    num_images = len(ldr_images)
    hdr_image = np.zeros((height, width, 3), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            E = np.zeros(3)
            for c in range(3): 
                valid = []
                weights = []
                for k in range(num_images):
                    pixel_value = ldr_images[k][i, j, c]
                    if 5 < pixel_value < 250:  
                        X = response_curve[pixel_value, c]  
                        weight = 1  
                        Eij = X / exposure_times[k]
                        valid.append(Eij)
                        weights.append(weight)
                if valid:
                    E[c] = np.average(valid, weights=weights)
                else:
                    E[c] = 0  
            hdr_image[i, j] = E

    return hdr_image

def reinhard_global_tone_mapping(hdr_image, alpha=0.18, delta=1e-6):
    luminance = 0.299*hdr_image[:,:,0] + 0.587*hdr_image[:,:,1] + 0.114*hdr_image[:,:,2]
    avg_luminance = np.exp(np.mean(np.log(luminance + 1e-8)))
    s_luminance = (alpha/avg_luminance) * luminance

    global_op = s_luminance / (1 + s_luminance)
    tone_mapped = np.empty(hdr_image.shape)
    tone_mapped[:,:,0] = global_op * (hdr_image[:,:,0]/luminance)
    tone_mapped[:,:,1] = global_op * (hdr_image[:,:,1]/luminance)
    tone_mapped[:,:,2] = global_op * (hdr_image[:,:,2]/luminance)
    tone_mapped[tone_mapped > 1] = 1
    tone_mapped *= 255

    return tonemapped.astype(np.uint8)

def main():

    # init
    curve = load_curve('curve.txt')
    images = load_images(image_paths)
    ldr_images = convert_images(images)

    hdr_image = create_hdr_image(ldr_images, exposure_times, curve)

    cv2.imwrite("output_hdr.jpg", cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR))
    tone_mapped_image = reinhard_global_tone_mapping(hdr_image)
    cv2.imwrite("output_tone_mapped.jpg", cv2.cvtColor(tone_mapped_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()