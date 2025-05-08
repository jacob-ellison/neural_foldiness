import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

from skimage import draw, measure
from skimage.filters import threshold_multiotsu

from scipy import ndimage
from scipy.ndimage import zoom, gaussian_filter1d



def read_image(image_path):
    """
    Reads an image (supports .jpg) and returns it as a PIL image

    Args:
        image_path (str or Path): Path to the image file.

    Returns:
        np.ndarray: Image as a PIL image.
    """
    image_path = Path(image_path)
    img_pil = Image.open(image_path)
    return img_pil
    
def bbox_area(contour):
    """
    Finds bounding bounding box area of a contour
    """
    _, _, w, h = cv2.boundingRect(contour)
    return w * h

def grayscale_to_intensity_xy(img):
    # Load image using PIL and convert to grayscale

    img = np.array(img)

    # Get image dimensions
    h, w = img.shape

    # Create meshgrid of x and y coordinates
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Flatten all arrays and stack into [intensity, x, y]
    intensities = img.flatten()
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    result = np.stack([intensities, x_flat, y_flat], axis=1)

def grayscale_to_distance_intensity_array(image):
    # Load the image as grayscale

    h, w = image.shape
    cx, cy = w // 2, h // 2

    result = []

    for y in range(h):
        for x in range(w):
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            intensity = image[y, x]
            result.append([distance, intensity])

    return np.array(result)


import cv2
import numpy as np

drawing = False
current_contour = []
mask = None
drawing_enabled = False  # To toggle drawing on and off

def mouse_callback(event, x, y, flags, param):
    global drawing, current_contour

    if event == cv2.EVENT_LBUTTONDOWN and drawing_enabled:
        drawing = True
        current_contour = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE and drawing_enabled:
        if drawing:
            current_contour.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP and drawing_enabled:
        drawing = False
        current_contour.append((x, y))

def draw_freehand_roi(image_path, save_mask_path="cortex_mask.png"):
    global current_contour, mask, drawing_enabled

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image.")

    clone = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.namedWindow("Draw Cortex ROI - 's' to Start/Stop, Enter to Save ROI, Esc to Finish")
    cv2.setMouseCallback("Draw Cortex ROI - 's' to Start/Stop, Enter to Save ROI, Esc to Finish", mouse_callback)

    while True:
        display = clone.copy()

        # Draw the current freehand path
        if len(current_contour) > 1:
            cv2.polylines(display, [np.array(current_contour)], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow("Draw Cortex ROI - 's' to Start/Stop, Enter to Save ROI, Esc to Finish", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # 's' to start/stop drawing
            drawing_enabled = not drawing_enabled
            print(f"Drawing {'enabled' if drawing_enabled else 'disabled'}")

        elif key == 13:  # Enter key
            if len(current_contour) > 2:
                cv2.fillPoly(mask, [np.array(current_contour)], 255)
                print(f"Saved ROI with {len(current_contour)} points.")
                current_contour = []

        elif key == 27:  # Esc key
            break

    cv2.destroyAllWindows()
    cv2.imwrite(save_mask_path, mask)
    print(f"Saved binary mask to: {save_mask_path}")
    return mask


def load_png_image(image_path):
    """
    Loads a PNG image from the specified path and returns it as a NumPy array.

    Parameters:
    - image_path (str): Path to the PNG image.

    Returns:
    - image (np.ndarray): Loaded image as a NumPy array.
    """
    # Load the PNG image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load in unchanged mode (including alpha channel)
    
    if image is None:
        raise ValueError(f"Error: Could not load the image at {image_path}. Please check the path.")
    
    # Check if the image has an alpha channel (transparency)
    if image.shape[2] == 4:
        print("Image has an alpha channel (transparency).")
    
    return image

import cv2
import numpy as np

class FreehandROITool:
    def __init__(self, image_path, save_mask_path="cortex_mask.png"):
        self.image_path = image_path
        self.save_mask_path = save_mask_path
        self.drawing = False
        self.drawing_enabled = False
        self.current_contour = []
        self.mask = None
        self.image = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.drawing_enabled:
            self.drawing = True
            self.current_contour = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_enabled and self.drawing:
            self.current_contour.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP and self.drawing_enabled:
            self.drawing = False
            self.current_contour.append((x, y))

    def draw(self):
        self.image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError("Could not load image.")

        clone = self.image.copy()
        self.mask = np.zeros_like(self.image)

        window_name = "Draw Cortex ROI - 's'=Start/Stop, Enter=Save ROI, Esc=Finish"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            display = clone.copy()

            if len(self.current_contour) > 1:
                cv2.polylines(display, [np.array(self.current_contour)], isClosed=False, color=(0, 255, 0), thickness=2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self.drawing_enabled = not self.drawing_enabled
                print(f"Drawing {'enabled' if self.drawing_enabled else 'disabled'}")

            elif key == 13:  # Enter key
                if len(self.current_contour) > 2:
                    fill_value = (255,) * self.mask.shape[2] if len(self.mask.shape) == 3 else 255
                    cv2.fillPoly(self.mask, [np.array(self.current_contour)], fill_value)
                    print(f"Saved ROI with {len(self.current_contour)} points.")
                    self.current_contour = []

            elif key == 27:  # Esc key
                break

        cv2.destroyAllWindows()
        cv2.imwrite(self.save_mask_path, self.mask)
        print(f"Saved binary mask to: {self.save_mask_path}")
        return self.mask

def extract_largest_blob(mask_array):
    """
    Extracts the largest connected component (blob) from a binary mask.

    Parameters:
        mask_array (np.ndarray): Binary image (2D or single-channel) where non-zero pixels are foreground.

    Returns:
        largest_blob_mask (np.ndarray): Binary mask of the largest blob (same shape as input).
    """

    # Ensure the input is a NumPy array
    if not isinstance(mask_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    # If it's color (3 channels), convert to grayscale
    if len(mask_array.shape) == 3:
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)

    # Convert to uint8 if needed
    if mask_array.dtype != np.uint8:
        mask_array = (mask_array > 0).astype(np.uint8) * 255

    # Threshold to make sure it's binary (0 or 255)
    _, bin_mask = cv2.threshold(mask_array, 1, 255, cv2.THRESH_BINARY)
    
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    if num_labels <= 1:
        # No components found (other than background)
        return np.zeros_like(binary_mask)

    # Find the label of the largest component (excluding background at index 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a new mask with only the largest blob
    largest_blob_mask = np.uint8(labels == largest_label) * 255

    return largest_blob_mask

def morph_open_close(binary_mask, kernel_size=3, iterations=1):
    """
    Apply morphological opening and closing to a binary mask.

    Parameters:
    - binary_mask: np.ndarray, binary mask with values 0 and 255 or 0 and 1
    - kernel_size: int, size of the square structuring element
    - iterations: int, number of times the operation is applied

    Returns:
    - opened_mask: np.ndarray, result after opening
    - closed_mask: np.ndarray, result after closing
    """
    # Ensure mask is uint8 type with values 0 or 255
    if binary_mask.max() == 1:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    else:
        binary_mask = binary_mask.astype(np.uint8)

    # Create a square kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply closing (dilation followed by erosion)
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # Apply opening (erosion followed by dilation)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)

   

    return opened


from scipy import ndimage

def keep_large_components(binary_mask, min_size=100):
    """
    Keeps connected components in the binary mask larger than `min_size`.

    Parameters:
    - binary_mask: np.ndarray, binary mask with values 0 and 1 or 0 and 255
    - min_size: int, minimum number of pixels for a component to be kept

    Returns:
    - filtered_mask: np.ndarray, binary mask with only large components
    """
    # Ensure binary format (0 and 1)
    mask = (binary_mask > 0).astype(np.uint8)

    # Label connected components
    labeled_mask, num_features = ndimage.label(mask)

    # Get sizes of components
    component_sizes = np.bincount(labeled_mask.ravel())

    # Zero out small components (label 0 is background)
    keep_labels = np.where(component_sizes >= min_size)[0]
    keep_labels = keep_labels[keep_labels != 0]  # exclude background

    # Create output mask
    filtered_mask = np.isin(labeled_mask, keep_labels).astype(np.uint8)

    return filtered_mask

def label_and_stack(data, gm_mask, wm_mask, cm_mask):
    # Gray matter (label = 0)
    masked_data = data[np.where(gm_mask)]
    gm = np.vstack((masked_data.T, 1 * np.ones(masked_data.shape[0]))).T

    # White matter (label = 1)
    masked_data = data[np.where(wm_mask)]
    wm = np.vstack((masked_data.T, 0 * np.ones(masked_data.shape[0]))).T

    # Center matter (label = 2)
    masked_data = data[np.where(cm_mask)]
    center = np.vstack((masked_data.T, 0 * np.ones(masked_data.shape[0]))).T

    # Combine all
    labeled_data = np.vstack((gm, wm, center))
    return labeled_data