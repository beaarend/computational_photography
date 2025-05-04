import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def build_gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv.pyrDown(img)
        gp.append(img.astype(np.float32))
    return gp

def build_laplacian_pyramid(gp):
    lp = []
    for i in range(len(gp) - 1):
        size = (gp[i].shape[1], gp[i].shape[0])
        GE = cv.pyrUp(gp[i + 1], dstsize=size)
        L = cv.subtract(gp[i], GE)
        lp.append(L)
    lp.append(gp[-1])  # Last level
    return lp

def blend_pyramids(lpA, lpB, gpM):
    blended = []
    for la, lb, gm in zip(lpA, lpB, gpM):
        gm = gm / 255.0 if gm.max() > 1 else gm  # normalize if necessary
        gm = cv.merge([gm, gm, gm])  # Make it 3-channel
        blended.append(la * gm + lb * (1 - gm))
    return blended

def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv.pyrUp(img, dstsize=size)
        img = cv.add(img, lp[i])
    return np.clip(img, 0, 255).astype(np.uint8)

def laplacian_blend(imgA, imgB, mask, levels=6):
    gpA = build_gaussian_pyramid(imgA, levels)
    gpB = build_gaussian_pyramid(imgB, levels)
    gpM = build_gaussian_pyramid(mask, levels)

    lpA = build_laplacian_pyramid(gpA)
    lpB = build_laplacian_pyramid(gpB)

    blended_pyramid = blend_pyramids(lpA, lpB, gpM)
    blended_image = reconstruct_from_laplacian(blended_pyramid)
    return blended_image

# ---- EXAMPLE USAGE ----
if __name__ == "__main__":
    imgA = cv.imread('images/ovo.png')
    imgB = cv.imread('images/pascoa.png')
    mask = cv.imread('images/mask_ovo.png', cv.IMREAD_GRAYSCALE)  # Binary mask

    if imgA is None or imgB is None or mask is None:
        raise FileNotFoundError("One or more input files not found. Check the paths.")

    # Ensure same size
    h, w = min(imgA.shape[0], imgB.shape[0]), min(imgA.shape[1], imgB.shape[1])
    imgA = cv.resize(imgA, (w, h))
    imgB = cv.resize(imgB, (w, h))
    mask = cv.resize(mask, (w, h))

    result = laplacian_blend(imgA, imgB, mask, levels=6)

    cv.imwrite("result_images/ovo_pascoa.png", result)  # Save the result

    cv.imshow("Blended", result)
    cv.waitKey(0)
    cv.destroyAllWindows()