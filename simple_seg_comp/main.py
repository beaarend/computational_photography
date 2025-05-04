import cv2 as cv
import numpy as np

def alpha_composite(foreground, background, alpha_mask):
    # Normalize alpha to [0, 1]
    alpha = alpha_mask.astype(np.float32) / 255.0
    alpha = cv.merge([alpha, alpha, alpha])  # convert to 3 channels
    # Composite: I = alpha * foreground + (1 - alpha) * background
    composite = cv.convertScaleAbs(alpha * foreground + (1 - alpha) * background)
    return composite

# Load foreground image and corresponding alpha mask
fg = cv.imread("images/GT04.png")
alpha = cv.imread("images/GT04_alpha.png", cv.IMREAD_GRAYSCALE)

# Load the selfie into which we want to insert the foreground
selfie = cv.imread("images/selfie.jpg")
if selfie is not None:
    # Let the user select a ROI on the selfie where the foreground should be placed.
    roi = cv.selectROI("Select ROI", selfie, fromCenter=False, showCrosshair=True)
    cv.destroyWindow("Select ROI")
    
    # roi is a tuple (x, y, w, h)
    x, y, w, h = roi
    
    # Resize the foreground and the alpha mask to match the ROI dimensions
    resized_fg = cv.resize(fg, (w, h))
    resized_alpha = cv.resize(alpha, (w, h))
    
    # Extract the region of interest (ROI) from the selfie
    roi_selfie = selfie[y:y+h, x:x+w]
    
    # Composite the resized foreground onto the ROI
    blended = alpha_composite(resized_fg, roi_selfie, resized_alpha)
    
    # Replace the ROI on the selfie with the blended result
    selfie[y:y+h, x:x+w] = blended
    
    # Save and display the result
    cv.imwrite("result_images/selfie_composite.png", selfie)
    cv.imshow("Selfie Composite", selfie)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Selfie image not found.")
