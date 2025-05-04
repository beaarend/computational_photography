import cv2 as cv
import numpy as np

drawing = False
value = cv.GC_FGD
colors = {cv.GC_BGD: (0, 0, 255), cv.GC_FGD: (0, 255, 0)}
drawing_img = None
mask = None
rect = None
rect_done = False

def draw_mask(event, x, y, flags, param):
    global drawing, drawing_img, value, mask
    if rect_done:
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            cv.circle(drawing_img, (x, y), 5, colors[value], -1)
            cv.circle(mask, (x, y), 5, value, -1)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False

def main():
    global drawing_img, mask, value, rect, rect_done

    img = cv.imread('images/duck.jpg')
    if img is None:
        raise FileNotFoundError("Image not found!")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    drawing_img = img.copy()

    # Step 1: Draw initial rectangle
    print("Draw a rectangle around the object and press ENTER.")

    rect = cv.selectROI("Select ROI", img, False, False)
    cv.destroyWindow("Select ROI")
    if rect[2] == 0 or rect[3] == 0:
        print("No rectangle selected.")
        return

    # Initialize with rectangle
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    rect_done = True

    # Create visual mask for user to refine
    drawing_img[:] = img
    mask_copy = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
    drawing_img[mask_copy == 255] = img[mask_copy == 255]  # Show segmented area

    # Set up for user refinement
    cv.namedWindow('input')
    cv.setMouseCallback('input', draw_mask)

    print("Refine with green (f) or red (b) scribbles, then press 'r' to re-run GrabCut, or 'q' to quit.")

    while True:
        cv.imshow('input', drawing_img)
        key = cv.waitKey(1) & 0xFF

        if key == ord('b'):
            value = cv.GC_BGD
            print("Drawing red (background).")
        elif key == ord('f'):
            value = cv.GC_FGD
            print("Drawing green (foreground).")
        elif key == ord('r'):
            # Re-run GrabCut with refined mask
            cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
            final_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')
            result = img * final_mask[:, :, np.newaxis]
            drawing_img[:] = result
        elif key == ord('q') or key == 27:
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
