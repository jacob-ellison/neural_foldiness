from src.utils import *
def compute_fractal_dimension(image):
    """
    Compute the fractal dimension of a 2D binary image using the box-counting method.

    Parameters:
        image (numpy.ndarray): Grayscale image (or binary edge map).

    Returns:
        float: Estimated fractal dimension.
    """
    # Step 1: Convert to binary if not already
    binary = (image > 0).astype(np.uint8) * 255

    # Step 2: Box-counting at multiple scales
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return np.sum(S > 0)

    # Ensure square image
    size = min(binary.shape)
    Z = binary[:size, :size]
    Z = Z > 0

    # Box sizes (powers of 2)
    sizes = 2 ** np.arange(1, int(np.log2(size))+1)
    counts = [boxcount(Z, s) for s in sizes]

    # Step 3: Linear fit to log-log plot
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]
    return fractal_dim



def compute_gyrification_index(image):
    """
    Compute a 2D Gyrification Index (GI) from a grayscale or binary brain image.

    Parameters:
        image (numpy.ndarray): Grayscale image of a coronal brain slice.

    Returns:
        float: Gyrification Index (GI) value.
    """
    # Step 1: Convert to binary if grayscale
    binary = (image > 0).astype(np.uint8) * 255

    # Step 2: Extract contours (assumes largest is cortex outline)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Use largest contour
    cortex_contour = max(contours, key=cv2.contourArea)
    cortex_length = cv2.arcLength(cortex_contour, closed=True)

    # Step 3: Compute convex hull of the contour
    hull = cv2.convexHull(cortex_contour)
    hull_length = cv2.arcLength(hull, closed=True)

    # Step 4: Compute GI
    if hull_length == 0:
        return 0.0
    gi = cortex_length / hull_length
    return gi


def bbox_area(contour):
    """
    Finds bounding bounding box area of a contour
    """
    _, _, w, h = cv2.boundingRect(contour)
    return w * h

def compute_avg_curvature_old(image):
    """
    Compute a 2D Average Curavature from a grayscale or binary brain image.

    Parameters:
        image (numpy.ndarray): Grayscale image of a coronal brain slice.

    Returns:
        float: average foldiness
    """

    mask_uint8 = (image.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    outer_contour = max(contours, key=bbox_area).squeeze()

    x = gaussian_filter1d(outer_contour[:, 0], sigma=5)
    y = gaussian_filter1d(outer_contour[:, 1], sigma=5)

    # Compute derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Compute curvature
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5
    curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    # Calculate average curvature (a proxy for "foldiness")
    average_curvature = np.mean(np.abs(curvature))
    return average_curvature , x, y

from scipy.ndimage import gaussian_filter1d

def compute_contour_curvature(contour, sigma=1):
    """
    Compute the average curvature of a single contour.
    """
    contour = contour.squeeze()
    if contour.ndim != 2 or contour.shape[0] < 5:
        return 0.0

    x = gaussian_filter1d(contour[:, 0], sigma=sigma)
    y = gaussian_filter1d(contour[:, 1], sigma=sigma)

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5
    curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    return np.mean(np.abs(curvature))


def compute_avg_curvature(image, area_thresh=100):
    """
    Compute 2D average curvature ("foldiness") from a grayscale or binary image.
    
    Returns:
        average_curvature (float): Mean curvature across all valid contours.
        x (np.ndarray): Smoothed x coordinates of the largest valid contour.
        y (np.ndarray): Smoothed y coordinates of the largest valid contour.
    """
    mask_uint8 = (image.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return 0.0, np.array([]), np.array([])

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_thresh]
    if not valid_contours:
        return 0.0, np.array([]), np.array([])

    curvatures = []
    for cnt in valid_contours:
        curv = compute_contour_curvature(cnt)
        curvatures.append(curv)

    # Get largest valid contour for x, y output
    largest = max(valid_contours, key=cv2.contourArea).squeeze()
    x = gaussian_filter1d(largest[:, 0], sigma=5)
    y = gaussian_filter1d(largest[:, 1], sigma=5)

    avg_curvature = np.mean(curvatures)
    return avg_curvature, x, y

def get_foldiness(img_path):
    # Read and preprocess image
    pil_image = read_image(img_path)
    np_image = np.array(pil_image.convert('L'))  # Convert to grayscale

    # Create a binary image where all non-white pixels are considered part of regions
    binary_img = np_image < 250

    # Find connected components
    labels = measure.label(binary_img, connectivity=1)
    regions = measure.regionprops(labels)

    # Identify and isolate the largest connected region
    largest_region = max(regions, key=lambda region: region.area)
    largest_blob_mask = labels == largest_region.label

    # Mask the original image to isolate the largest blob
    masked_image = np_image * largest_blob_mask

    # Apply Multi-Otsu thresholding
    thresholds = threshold_multiotsu(masked_image, classes=4)

    # Create a refined mask by thresholding between selected classes
    refined_mask = (masked_image > thresholds[1]) & (masked_image < thresholds[-1]) & (masked_image > 1)

    # Morphological filtering to smooth and clean the mask
    structure = ndimage.generate_binary_structure(2, 2)
    closed = ndimage.binary_closing(refined_mask, structure=structure, iterations=2)
    opened = ndimage.binary_opening(closed, structure=structure, iterations=2)

    # Re-label and isolate the largest component after filtering
    labels = measure.label(opened, connectivity=1)
    regions = measure.regionprops(labels)
    largest_region = max(regions, key=lambda region: region.area)
    largest_blob_mask_gm = labels == largest_region.label

    # Compute the metrics
    GI = compute_gyrification_index(largest_blob_mask_gm)
    FD = compute_fractal_dimension(largest_blob_mask_gm)
    average_curvature, x, y = compute_avg_curvature_old(largest_blob_mask_gm)

    # Compute Foldiness
    foldiness = (4*average_curvature) + (.5*GI) + (.2*FD)

    # Visualization
    plt.imshow(masked_image, cmap='gray')
    plt.plot(x, y, color='blue', linewidth=2)
    plt.axis('off')

    # Overlay metrics on image
    plt.text(binary_img.shape[0]*.1, binary_img.shape[1]*.1, f"Avg curvature: {average_curvature:.4f}", color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
    plt.text(binary_img.shape[0]*.1, binary_img.shape[1]*.2, f"Fractal Dimension: {FD:.4f}", color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
    plt.text(binary_img.shape[0]*.1, binary_img.shape[1]*.3, f"Gyrification Index: {GI:.4f}", color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
    plt.text(binary_img.shape[0]*.1, binary_img.shape[1]*.4, f"Foldiness Score: {foldiness:.4f}", color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))

    plt.show()

    return average_curvature, FD, GI, foldiness


def get_foldiness_from_mask(img_path,mask_man):
    # Read and preprocess image
    pil_image = read_image(img_path)
    np_image = np.array(pil_image.convert('L'))  # Convert to grayscale

    binary_img = np_image < 255

    # Mask the original image to isolate the largest blob

    # structure = ndimage.generate_binary_structure(2, 2)
    # closed = ndimage.binary_closing(mask_man, structure=structure, iterations=2)
    # opened = ndimage.binary_opening(closed, structure=structure, iterations=2)

    # Re-label and isolate the largest component after filtering
    labels = measure.label(mask_man, connectivity=1)
    regions = measure.regionprops(labels)
    largest_region = max(regions, key=lambda region: region.area)
    largest_blob_mask_gm = labels == largest_region.label

    # Compute the metrics
    GI = compute_gyrification_index(mask_man)
    FD = compute_fractal_dimension(mask_man)
    average_curvature, x, y = compute_avg_curvature(mask_man)

    # Compute Foldiness
    foldiness = (4*average_curvature) + (.5*GI) + (.2*FD)

    # Visualization
    plt.imshow(np_image, cmap='gray')
    plt.plot(x, y, color='blue', linewidth=2)
    plt.axis('off')

    # Overlay metrics on image
    plt.text(binary_img.shape[0]*.1, binary_img.shape[1]*.1, f"Avg curvature: {average_curvature:.4f}", color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
    plt.text(binary_img.shape[0]*.1, binary_img.shape[1]*.2, f"Fractal Dimension: {FD:.4f}", color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
    plt.text(binary_img.shape[0]*.1, binary_img.shape[1]*.3, f"Gyrification Index: {GI:.4f}", color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
    plt.text(binary_img.shape[0]*.1, binary_img.shape[1]*.4, f"Foldiness Score: {foldiness:.4f}", color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))

    plt.show()

    return average_curvature, FD, GI, foldiness