#!/usr/bin/env python3
"""
trichome_pipeline.py
=====================

This script provides an end‑to‑end workflow for manual annotation and
morphological analysis of plant trichomes.  It allows you to load raw
image files (JPEG, PNG, TIFF), interactively label individual
trichomes as polygons, and then extract a suite of quantitative
features from each labelled instance.  The extracted features cover
basic shape descriptors (area, perimeter, aspect ratio), centreline
statistics (length, tortuosity, curvature) and simple heuristics
for classifying the trichome into broad morphotypes (unbranched,
branched, capitate or peltate).

Key features:
 - Supports TIFF images with multiple pages via Pillow.
 - Interactive polygon annotation using matplotlib.  Left click adds
   vertices, right click closes the current polygon.  Press **n** to
   begin a new polygon and **q** to finish annotation.
 - Skeletonisation implemented from first principles to avoid
   external dependencies.  Endpoints and branch points are detected
   from the thinned skeleton.
 - Centreline extraction along the longest geodesic between
   endpoints.  Width profiles are obtained from a distance transform
   of the mask.
 - Heuristic rules to assign a coarse morphotype and subtype based on
   branching and head size.  These rules can be refined or replaced
   with a machine‑learning model if you supply training data.
 - Outputs per‑instance feature tables (CSV) and optional quality
   control (QC) panels illustrating the mask, skeleton and width
   profile.  QC panels are useful for verifying that feature
   extraction behaves sensibly on your data.

Usage examples:
```
# annotate a single image and compute features
python trichome_pipeline.py --image 250306_HT_11.jpg --pixel_size_um 0.5 --outdir out

# analyse an image using a pre‑existing instance mask
python trichome_pipeline.py --image 250529_LC_015_2.tif --mask 250529_LC_015_2_inst.png \
    --pixel_size_um 0.2 --outdir out

# process all images in a folder (annotate missing masks)
python trichome_pipeline.py --input_dir my_images --pixel_size_um 0.1 --outdir out
```

The script is designed to be self contained and runs on Python 3.  It
uses only standard library modules plus OpenCV, NumPy, Matplotlib,
Pillow and SciPy (all of which are available in the execution
environment).  No external network access is required.

"""

import argparse
import os
import sys
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image, ImageSequence
import matplotlib
import matplotlib.pyplot as plt
import hashlib
import scipy.ndimage as ndi

# Use the existing matplotlib backend.  Do not force a GUI backend here
# since this script may be executed in environments without Tk.  The
# default backend will automatically switch to a non‑interactive one
# (e.g. Agg) when run headlessly.  Users who wish to annotate
# interactively should run this script in an environment with a GUI
# backend available.


def load_image(path: str, page: Optional[int] = None) -> np.ndarray:
    """Load an image from disk.  Supports multi‑page TIFF via Pillow.

    Args:
        path: Path to the image file (JPEG, PNG, TIFF).
        page: Optional page index for multi‑page TIFFs.  If None,
              the first page is returned.

    Returns:
        image: NumPy array in grayscale (uint8).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file '{path}' does not exist.")
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.tif', '.tiff'}:
        img = Image.open(path)
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            if page is not None and page < img.n_frames:
                img.seek(page)
            else:
                # default to first page
                img.seek(0)
        arr = np.array(img)
        # Convert 16‑bit to 8‑bit if necessary
        if arr.dtype == np.uint16:
            arr = (arr / 256).astype(np.uint8)
    else:
        # Use OpenCV for non‑TIFF for speed
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"Failed to load image '{path}'")
    return arr


def annotate_image(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Interactively annotate trichome regions in an image.

    A matplotlib window is opened showing the image.  You can draw
    multiple polygons to label trichomes.  Left click to add points,
    right click to close the current polygon.  Press **n** to begin a
    new polygon; press **q** when you are done.  Press **x** to skip
    annotation entirely and discard this image.  After finishing,
    a mask image with unique integer labels per instance is returned.

    Args:
        image: Input image (grayscale or RGB) as a NumPy array.

    Returns:
        mask: 2D array of integers where background=0 and each
              annotated trichome region has a unique value ≥1.  If
              the user pressed 'x' to skip, this mask will be None.
        skipped: Boolean indicating whether the user chose to skip
              this image (by pressing 'x').
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Annotate trichomes: left click=add point, right click=close polygon,\n'
                 'n=start new, q=finish, x=skip')
    polygons: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    finished = False
    skip_image = False

    def on_click(event):
        nonlocal current
        # Only register clicks inside the axes
        if event.inaxes != ax:
            return
        if event.button == 1:  # left click
            current.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=4)
            fig.canvas.draw()
        elif event.button == 3:  # right click closes polygon
            if current:
                polygons.append(current.copy())
                # draw the polygon outline
                poly = np.array(current)
                ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), 'r-')
                current.clear()
                fig.canvas.draw()

    def on_key(event):
        nonlocal current, finished, skip_image
        if event.key == 'n':
            # start a new polygon if current has points
            if current:
                polygons.append(current.copy())
                poly = np.array(current)
                ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), 'r-')
                current.clear()
                fig.canvas.draw()
        elif event.key == 'q':
            # finish annotation
            if current:
                polygons.append(current.copy())
                poly = np.array(current)
                ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), 'r-')
                current.clear()
            finished = True
            plt.close(fig)
        elif event.key == 'x':
            # skip this image completely
            skip_image = True
            # no need to close polygons or store current points
            finished = True
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    # If skipped, return None and skip flag
    if skip_image:
        return None, True
    # Create a mask from polygons
    mask = np.zeros(image.shape[:2], dtype=np.int32)
    for idx, poly in enumerate(polygons, start=1):
        pts = np.array(poly, np.int32)
        cv2.fillPoly(mask, [pts], idx)
    return mask, False


def skeletonize(bin_img: np.ndarray) -> np.ndarray:
    """Compute a one‑pixel thick skeleton of a binary image.

    This function implements a classic morphological skeletonisation
    algorithm using erosions and dilations with a cross
    structuring element.  It iteratively erodes the foreground and
    accumulates the boundary pixels that would disappear after
    erosion until nothing remains.

    Args:
        bin_img: Binary image (dtype uint8, 0 background, >0 foreground).

    Returns:
        skel: Binary image containing the skeleton (0 background, 1
              foreground).  The skeleton will be a subset of the
              original foreground pixels.
    """
    # Ensure input is binary 0/1
    img = (bin_img > 0).astype(np.uint8)
    skel = np.zeros_like(img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def calibrate_image(image: np.ndarray) -> float:
    """Interactively calibrate pixel size using a scale bar on the image.

    This function opens a window showing the image and asks the user
    to click two points along a known scale bar (e.g. the endpoints of
    a 30 µm marker).  It then prompts the user to enter the real
    length of the scale bar in micrometres.  The pixel size is
    computed as real_length_um divided by the distance between the
    clicked points in pixels.

    Args:
        image: The image array (grayscale or RGB) on which to perform
            calibration.

    Returns:
        pixel_size_um: The pixel size in micrometres per pixel derived
            from the user input.

    Raises:
        ValueError: If calibration points are not properly selected.
    """
    import matplotlib.pyplot as plt
    points: List[Tuple[float, float]] = []
    fig, ax = plt.subplots(figsize=(8, 8))
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        # convert BGR to RGB if needed
        if image.shape[2] == 3:
            b, g, r = cv2.split(image)
            ax.imshow(cv2.merge([r, g, b]))
        else:
            ax.imshow(image)
    ax.set_title('Calibration: click two points along the scale bar')
    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=4)
            fig.canvas.draw()
            if len(points) == 2:
                plt.close(fig)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    if len(points) != 2:
        raise ValueError("Exactly two points must be selected for calibration.")
    (x0, y0), (x1, y1) = points
    pixel_length = np.hypot(x1 - x0, y1 - y0)
    while True:
        try:
            real_length_str = input("Enter the real-world length of the drawn line (in µm): ")
            real_length_um = float(real_length_str.strip())
            if real_length_um <= 0:
                print("Please enter a positive number for the real length.")
                continue
            break
        except ValueError:
            print("Invalid number. Please enter the real length in micrometres.")
    pixel_size_um = real_length_um / pixel_length
    print(f"Calibration complete: {pixel_size_um:.6f} µm per pixel")
    return pixel_size_um


def find_skeleton_endpoints_and_branches(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Identify endpoints and branch points in a skeleton image.

    An endpoint is a skeleton pixel with exactly one 8‑connected
    neighbour; a branch point has more than two 8‑connected neighbours.

    Args:
        skel: Binary skeleton image (dtype uint8, values 0 or 1).

    Returns:
        endpoints: List of (row, col) coordinates of skeleton endpoints.
        branchpoints: List of (row, col) coordinates of branch points.
    """
    endpoints: List[Tuple[int, int]] = []
    branches: List[Tuple[int, int]] = []
    skel_padded = np.pad(skel, 1, mode='constant', constant_values=0)
    rows, cols = skel.shape
    for y in range(rows):
        for x in range(cols):
            if skel[y, x] == 0:
                continue
            # extract 3x3 neighbourhood (padded) to count neighbours
            nbhd = skel_padded[y:y + 3, x:x + 3]
            num_neigh = int(nbhd.sum()) - 1  # exclude centre pixel
            if num_neigh == 1:
                endpoints.append((y, x))
            elif num_neigh > 2:
                branches.append((y, x))
    return endpoints, branches


def geodesic_path(skel: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Compute a path between two points on the skeleton.

    This function performs an unweighted breadth‑first search over
    8‑connected neighbours in the skeleton and reconstructs the path
    between the specified start and end pixels.  If no path exists,
    an empty list is returned.

    Args:
        skel: Binary skeleton image (uint8 0/1) to navigate.
        start: (row, col) coordinates of the starting endpoint.
        end: (row, col) coordinates of the target endpoint.

    Returns:
        path: List of (row, col) coordinates representing the path
              from start to end inclusive.  Returns empty list if
              endpoints are not connected.
    """
    from collections import deque
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    q = deque()
    q.append(start)
    visited[start] = True
    found = False
    # offsets for 8‑connected neighbourhood
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while q:
        cy, cx = q.popleft()
        if (cy, cx) == end:
            found = True
            break
        for dy, dx in offsets:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and skel[ny, nx] > 0:
                visited[ny, nx] = True
                parent[(ny, nx)] = (cy, cx)
                q.append((ny, nx))
    if not found:
        return []
    # reconstruct path
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def extract_centerline_and_width(mask: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float], List[float], List[int], int]:
    """Extract the longest centreline from a binary mask and measure widths.

    The function skeletonises the mask, finds endpoints and branch
    points, and then determines the longest geodesic path between any
    pair of endpoints.  It uses a distance transform to estimate the
    local radius at each skeleton pixel and thereby computes a width
    profile along the path.  It returns the path coordinates, list
    of diameters (2*distance), list of curvature values (radians) and
    list of endpoints, plus the number of branch points.

    Args:
        mask: Binary mask of a single trichome instance (dtype
              uint8 with values 0/255).

    Returns:
        path: Ordered list of (row, col) coordinates for the
              extracted centreline.
        diameters: List of diameters (in pixels) corresponding to
              each coordinate in the path.
        curvatures: List of signed curvature estimates (radians) for
              each interior point along the path (first and last
              entries are 0.0 by convention).
        endpoints: List of endpoints found in the skeleton.  At
              least two endpoints must exist to define a path.
        num_branches: Number of branch points detected.
    """
    bin_mask = (mask > 0).astype(np.uint8)
    # Perform simple morphological cleaning to fill small holes and remove tiny gaps.
    # This helps stabilise skeletonisation and centreline extraction.
    try:
        # Apply a small opening followed by closing to remove speckle noise and fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    except Exception:
        # If morphological operations fail (e.g. empty mask), proceed without cleaning
        pass
    skel = skeletonize(bin_mask)
    endpoints, branches = find_skeleton_endpoints_and_branches(skel)
    # If fewer than 2 endpoints, return empty structures
    if len(endpoints) < 2:
        return [], [], [], endpoints, len(branches)
    # Compute distance transform for width estimation
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    # Find the pair of endpoints with the longest path (by number of steps)
    best_path: List[Tuple[int, int]] = []
    max_len = -1
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p1 = endpoints[i]
            p2 = endpoints[j]
            path = geodesic_path(skel, p1, p2)
            if path:
                if len(path) > max_len:
                    max_len = len(path)
                    best_path = path
    # If no path found, return default
    if not best_path:
        return [], [], [], endpoints, len(branches)
    # Compute diameter and curvature along the path
    diameters: List[float] = []
    curvatures: List[float] = []
    for idx, (y, x) in enumerate(best_path):
        diam = dist[y, x] * 2.0
        diameters.append(float(diam))
        if 0 < idx < len(best_path) - 1:
            # compute discrete curvature via turn angle
            y0, x0 = best_path[idx - 1]
            y1, x1 = best_path[idx]
            y2, x2 = best_path[idx + 1]
            v1 = np.array([x1 - x0, y1 - y0], dtype=np.float64)
            v2 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            # avoid division by zero
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                # compute angle between vectors
                dot = np.dot(v1, v2) / (norm1 * norm2)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                # sign of curvature: positive if turning left, negative if right
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                signed_angle = angle if cross > 0 else -angle
            else:
                signed_angle = 0.0
            curvatures.append(signed_angle)
        else:
            curvatures.append(0.0)
    return best_path, diameters, curvatures, endpoints, len(branches)


def compute_features(mask: np.ndarray, pixel_size_um: float) -> Dict[str, float]:
    """Compute morphological features and assign a morphotype.

    This function wraps centreline extraction and then summarises a
    variety of quantitative descriptors: length, straightness,
    tortuosity, curvature statistics, width statistics and branch
    counts.  It applies heuristic rules to classify the trichome
    broadly into unbranched, branched or glandular types and to
    assign subtypes (straight, hooked, capitate, peltate).  If the
    path extraction fails (e.g. too few endpoints) the function
    returns a minimal feature set with NaNs.

    Args:
        mask: Binary mask for a single trichome instance (uint8
              values 0/255).
        pixel_size_um: Physical pixel size (µm per pixel).  This
              converts pixel measurements into micrometres.

    Returns:
        features: Dictionary mapping feature names to values.  All
            length‑based metrics are reported in micrometres.
    """
    # Compute area and perimeter in pixel units
    bin_mask = (mask > 0).astype(np.uint8)
    area_px = float(cv2.countNonZero(bin_mask))
    # perimeter using OpenCV findContours; handle small masks
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_px = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    # Basic shape descriptors
    features: Dict[str, float] = {}
    features['area_um2'] = area_px * (pixel_size_um ** 2)
    features['perimeter_um'] = perimeter_px * pixel_size_um
    # bounding box
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0:
        # empty mask; return NaNs
        for key in ['length_um', 'straight_length_um', 'tortuosity', 'width_base_um',
                    'width_tip_um', 'width_max_um', 'curvature_mean_rad',
                    'curvature_max_rad', 'curvature_sum_rad', 'num_branchpoints']:
            features[key] = float('nan')
        features['morphotype'] = 'unknown'
        features['subtype'] = 'unknown'
        return features
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox_width_px = x_max - x_min + 1
    bbox_height_px = y_max - y_min + 1
    features['bbox_aspect_ratio'] = float(bbox_width_px) / float(bbox_height_px) if bbox_height_px > 0 else float('nan')
    # Fit an ellipse if enough contour points exist
    ellipse_major_um = float('nan')
    ellipse_minor_um = float('nan')
    ellipse_angle_deg = float('nan')
    if contours and len(contours[0]) >= 5:
        (xc, yc), (d1, d2), angle = cv2.fitEllipse(contours[0])
        # major axis = max(d1,d2)
        major = max(d1, d2)
        minor = min(d1, d2)
        ellipse_major_um = major * pixel_size_um
        ellipse_minor_um = minor * pixel_size_um
        ellipse_angle_deg = float(angle)
    features['ellipse_major_um'] = ellipse_major_um
    features['ellipse_minor_um'] = ellipse_minor_um
    features['ellipse_angle_deg'] = ellipse_angle_deg

    # Compute convex hull area for contraction ratio
    # Use the largest contour for hull computation
    contraction_ratio = float('nan')
    try:
        if contours and len(contours[0]) >= 3:
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                contraction_ratio = 1.0 - (area_px / hull_area)
    except Exception:
        contraction_ratio = float('nan')
    features['contraction_ratio'] = contraction_ratio
    # Skeleton and width/curvature
    path, diameters, curvatures, endpoints, num_branches = extract_centerline_and_width(bin_mask)
    features['num_endpoints'] = len(endpoints)
    features['num_branchpoints'] = num_branches
    if not path:
        # cannot compute centreline features
        for key in ['length_um', 'straight_length_um', 'tortuosity', 'width_base_um',
                    'width_tip_um', 'width_max_um', 'curvature_mean_rad',
                    'curvature_max_rad', 'curvature_sum_rad']:
            features[key] = float('nan')
        # classify as unknown
        features['morphotype'] = 'unknown'
        features['subtype'] = 'unknown'
        return features
    # Convert centreline coordinates to physical units and compute length
    length_px = 0.0
    for i in range(len(path) - 1):
        y0, x0 = path[i]
        y1, x1 = path[i + 1]
        length_px += np.hypot(x1 - x0, y1 - y0)
    length_um = length_px * pixel_size_um
    # Straight line distance between endpoints
    y_start, x_start = path[0]
    y_end, x_end = path[-1]
    straight_px = np.hypot(x_end - x_start, y_end - y_start)
    straight_um = straight_px * pixel_size_um
    tortuosity = (length_px / straight_px) if straight_px > 0 else float('nan')
    # Width statistics (in micrometres)
    diameters_um = [d * pixel_size_um for d in diameters]
    width_base_um = np.mean(diameters_um[: max(1, len(diameters_um) // 10)])  # average over first 10%
    width_tip_um = np.mean(diameters_um[-max(1, len(diameters_um) // 10):])  # last 10%
    width_max_um = max(diameters_um) if diameters_um else float('nan')
    # Curvature statistics (radians)
    curvatures_abs = [abs(c) for c in curvatures]
    curvature_mean = float(np.mean(curvatures_abs))
    curvature_max = float(max(curvatures_abs))
    curvature_sum = float(np.sum(curvatures_abs))
    # Compute skeleton length and ratio metrics.  Skeleton length is the total
    # number of skeleton pixels times pixel size.  Skeleton ratio compares
    # skeleton length to centreline length.
    try:
        skel = skeletonize(bin_mask)
        skel_px = float(np.count_nonzero(skel))
        skeleton_length_um = skel_px * pixel_size_um
        skeleton_ratio = (skeleton_length_um / length_um) if length_um > 0 else float('nan')
    except Exception:
        skeleton_length_um = float('nan')
        skeleton_ratio = float('nan')
    features.update({
        'length_um': length_um,
        'straight_length_um': straight_um,
        'tortuosity': tortuosity,
        'width_base_um': width_base_um,
        'width_tip_um': width_tip_um,
        'width_max_um': width_max_um,
        'curvature_mean_rad': curvature_mean,
        'curvature_max_rad': curvature_max,
        'curvature_sum_rad': curvature_sum,
        'skeleton_length_um': skeleton_length_um,
        'skeleton_ratio': skeleton_ratio
    })
    # Heuristic classification into morphotype and subtype
    morphotype = 'unknown'
    subtype = 'unknown'
    # Branching check
    if num_branches > 0:
        morphotype = 'branched'
        subtype = 'branched'
    else:
        # width ratio for head detection
        head_ratio = width_max_um / width_base_um if width_base_um > 0 else 1.0
        if head_ratio > 2.0:
            morphotype = 'glandular'
            # differentiate capitate vs peltate by head dominance
            if head_ratio > 4.0:
                subtype = 'peltate'
            else:
                subtype = 'capitate'
        else:
            morphotype = 'unbranched'
            # straight vs hooked via curvature
            if curvature_max > 0.4:  # threshold in radians (~23°)
                subtype = 'hooked'
            else:
                subtype = 'straight'
    features['morphotype'] = morphotype
    features['subtype'] = subtype
    return features


def save_qc_panels(image: np.ndarray, mask: np.ndarray, path: List[Tuple[int, int]], diameters: List[float], curvatures: List[float], out_path_prefix: str) -> None:
    """Save quality control panels illustrating the processing steps.

    Three panels are produced: the raw image with mask overlay, the
    skeleton and centreline, and the width/curvature profiles along
    the centreline.  Panels are saved as PNG files with the given
    prefix followed by a descriptive suffix.

    Args:
        image: Raw image array (grayscale or RGB).
        mask: Binary instance mask (uint8, values 0/255).
        path: List of (row, col) coordinates for the centreline.
        diameters: Width profile (in pixels) along the centreline.
        curvatures: Curvature values (radians) along the centreline.
        out_path_prefix: File path prefix (without extension) for
            output images.
    """
    import matplotlib.pyplot as plt  # import here to avoid top‑level dependency
    # Panel 1: raw image with mask overlay
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Convert grayscale to RGB for overlay
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image.copy()
    # overlay mask in red
    overlay = rgb.copy().astype(float)
    # normalise mask values to [0,1]
    norm_mask = (mask > 0).astype(float)
    overlay[:, :, 0] = overlay[:, :, 0] * (1 - 0.5 * norm_mask) + 255 * (0.5 * norm_mask)
    overlay[:, :, 1] = overlay[:, :, 1] * (1 - 0.5 * norm_mask)
    overlay[:, :, 2] = overlay[:, :, 2] * (1 - 0.5 * norm_mask)
    axes[0].imshow(overlay.astype(np.uint8))
    axes[0].set_title('Raw + Mask')
    axes[0].axis('off')
    # Panel 2: skeleton and centreline
    skel = skeletonize((mask > 0).astype(np.uint8))
    axes[1].imshow(skel, cmap='gray')
    if path:
        ys, xs = zip(*path)
        axes[1].plot(xs, ys, 'r-', linewidth=1)
        axes[1].plot(xs[0], ys[0], 'go', markersize=4)  # start (base)
        axes[1].plot(xs[-1], ys[-1], 'bo', markersize=4)  # end (tip)
    axes[1].set_title('Skeleton & Centreline')
    axes[1].axis('off')
    # Panel 3: width and curvature profiles
    if diameters and curvatures:
        ax3 = axes[2]
        t = np.arange(len(diameters))
        ax3.plot(t, diameters, 'r-', label='Width (px)')
        ax3.set_xlabel('Along‑path index')
        ax3.set_ylabel('Width (px)', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        ax3.set_title('Width & Curvature')
        # secondary axis for curvature
        ax4 = ax3.twinx()
        ax4.plot(t, curvatures, 'b--', label='Curvature (rad)')
        ax4.set_ylabel('Curvature (rad)', color='b')
        ax4.tick_params(axis='y', labelcolor='b')
    axes[2].grid(True)
    plt.tight_layout()
    fig.savefig(out_path_prefix + '_qc.png', dpi=150)
    plt.close(fig)


def process_single_image(img_path: str, mask_path: Optional[str], pixel_size_um: float, outdir: str,
                         page: Optional[int] = None, qc: bool = True, subdir: Optional[str] = None,
                         calibrate: bool = False, auto_annotation: bool = False,
                         lo_diff: int = 20, hi_diff: int = 20, auto_polygon: bool = False,
                         polygon_method: str = 'otsu', edge_mode: str = 'single') -> List[Dict[str, float]]:
    """Process a single image: annotate or load masks, extract features, save QC.

    Args:
        img_path: Path to the raw image file.
        mask_path: Optional path to an instance mask.  If None, you
            will be prompted to annotate via interactive GUI.
        pixel_size_um: Pixel size in micrometres (µm per pixel).
        outdir: Directory to write outputs.  CSVs and QC images are
            saved here.
        page: If the image is a multi‑page TIFF, specify which page
            to load (0‑based).  Ignored for non‑TIFF images.
        qc: Whether to save QC panels for each instance.
        auto_annotation: If True, use automatic flood‑fill segmentation instead of manual polygon drawing.
        lo_diff: Lower intensity tolerance for flood fill in automatic annotation.
        hi_diff: Upper intensity tolerance for flood fill in automatic annotation.
        auto_polygon: If True, use polygon-drawn ROI with Otsu-based segmentation instead of point-based auto annotation or manual drawing.

    Returns:
        feature_list: List of dictionaries containing features and
            metadata for each annotated trichome instance.
    """
    # Load raw image
    img = load_image(img_path, page)
    # Determine base name for outputs
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    if page is not None:
        base_name += f'_page{page}'
    # Determine the directory where outputs for this image will be saved.  If a
    # relative subdirectory is provided (from a recursive scan), replicate
    # that structure under the main outdir.  Otherwise, use outdir directly.
    target_outdir = os.path.join(outdir, subdir) if subdir else outdir

    # Load or create mask
    if mask_path is None:
        print(f"Annotating image {img_path}")
        # Choose annotation method based on flags
        if auto_polygon:
            mask, skipped = annotate_image_auto_polygon(img, method=polygon_method, edge_mode=edge_mode)
        elif auto_annotation:
            mask, skipped = annotate_image_auto(img, lo_diff=lo_diff, hi_diff=hi_diff)
        else:
            mask, skipped = annotate_image(img)
        if skipped:
            # Do not create a mask or process features for this image
            return []
        # Save mask for later reuse in the target output directory
        os.makedirs(target_outdir, exist_ok=True)
        mask_save_path = os.path.join(target_outdir, base_name + '_inst.png')
        cv2.imwrite(mask_save_path, mask.astype(np.uint16))
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to load mask '{mask_path}'")
    # Ensure the target output directory exists
    os.makedirs(target_outdir, exist_ok=True)
    # If dualcurve edge mode is requested and this image was annotated manually (mask_path is None),
    # allow the user to refine each instance segmentation by selecting tip/root points.  We perform
    # this refinement before iterating over instances.  This avoids nested event loops.
    # Always initialize a boundary metrics map to store per-instance curvature metrics.
    boundary_metrics_map: Dict[int, Dict[str, float]] = {}
    if edge_mode == 'dualcurve' and mask_path is None:
        unique_ids = [i for i in np.unique(mask) if i > 0]
        # Prepare a new mask to hold refined instances
        refined_mask_all = np.zeros_like(mask, dtype=np.int32)
        new_id_counter = 1
        # Dictionary to store boundary curvature metrics keyed by new instance ID
        boundary_metrics_map: Dict[int, Dict[str, float]] = {}
        for inst in unique_ids:
            inst_mask_bool = (mask == inst)
            # Invoke interactive dualcurve refinement for this instance
            try:
                refined_inst, metrics = annotate_instance_dualcurve(img, inst_mask_bool)
            except Exception:
                # On exception, fallback to original instance and empty metrics
                refined_inst = inst_mask_bool.copy()
                metrics = {
                    'upper_curv_mean_rad': float('nan'),
                    'upper_curv_max_rad': float('nan'),
                    'upper_curv_sum_rad': float('nan'),
                    'lower_curv_mean_rad': float('nan'),
                    'lower_curv_max_rad': float('nan'),
                    'lower_curv_sum_rad': float('nan')
                }
            if refined_inst is not None and np.any(refined_inst):
                refined_mask_all[refined_inst] = new_id_counter
                # Store metrics for this new instance id
                boundary_metrics_map[new_id_counter] = metrics
                new_id_counter += 1
        # Replace mask with refined mask
        mask = refined_mask_all
    else:
        # No dualcurve refinement; leave mask unchanged and metrics map empty
        pass

    # Iterate over instances in mask
    feature_list: List[Dict[str, float]] = []
    instance_ids = np.unique(mask)
    instance_ids = [i for i in instance_ids if i > 0]
    # Determine pixel size for this image.  If calibrate flag is set, invoke
    # interactive calibration; otherwise use the provided pixel_size_um.
    current_pixel_size = pixel_size_um
    if calibrate:
        try:
            current_pixel_size = calibrate_image(img)
        except Exception as e:
            print(f"Calibration failed for image {img_path}: {e}. Falling back to provided pixel size.")
            current_pixel_size = pixel_size_um
    for inst_id in instance_ids:
        inst_mask = (mask == inst_id).astype(np.uint8)
        features = compute_features(inst_mask, current_pixel_size)
        # Add metadata
        features['image'] = base_name
        features['instance_id'] = int(inst_id)
        # Include subdirectory information for traceability
        features['subdir'] = subdir if subdir else ''
        # If dualcurve metrics are available, attach them; otherwise fill with NaNs
        bmetrics = boundary_metrics_map.get(int(inst_id)) if 'boundary_metrics_map' in locals() else None
        # Define default NaN metrics
        default_metrics = {
            'upper_curv_mean_rad': float('nan'),
            'upper_curv_max_rad': float('nan'),
            'upper_curv_sum_rad': float('nan'),
            'lower_curv_mean_rad': float('nan'),
            'lower_curv_max_rad': float('nan'),
            'lower_curv_sum_rad': float('nan')
        }
        if bmetrics:
            # ensure all metrics keys exist; merge with defaults
            for key, val in default_metrics.items():
                features[key] = bmetrics.get(key, val)
        else:
            # No metrics provided; insert defaults
            for key, val in default_metrics.items():
                features[key] = val
        feature_list.append(features)
        # Print a concise summary of features so the user can see results immediately
        try:
            length_val = features.get('length_um', float('nan'))
            width_max_val = features.get('width_max_um', float('nan'))
            curv_max_val = features.get('curvature_max_rad', float('nan'))
            morph = features.get('morphotype', 'unknown')
            subtype = features.get('subtype', 'unknown')
            print(f"Instance {inst_id} summary: morphotype={morph}, subtype={subtype}, "
                  f"length={length_val:.2f} µm, max width={width_max_val:.2f} µm, "
                  f"max curvature={curv_max_val:.2f} rad")
        except Exception:
            # In case of NaNs or formatting issues, fallback to printing the raw dict
            print(f"Instance {inst_id} features: {features}")
        # Generate QC panels if requested
        if qc:
            path, diameters, curvatures, _, _ = extract_centerline_and_width(inst_mask)
            qc_prefix = os.path.join(target_outdir, f"{base_name}_inst{inst_id}")
            save_qc_panels(img, inst_mask * 255, path, diameters, curvatures, qc_prefix)
    return feature_list


def main():
    parser = argparse.ArgumentParser(description="Trichome annotation and analysis pipeline")
    parser.add_argument('--image', type=str, help="Path to a single image file (JPEG/PNG/TIFF)")
    parser.add_argument('--mask', type=str, default=None, help="Path to instance mask (PNG/TIFF)")
    parser.add_argument('--input_dir', type=str, help="Directory containing images to process")
    parser.add_argument('--pixel_size_um', type=float, help="Pixel size in micrometres (µm per pixel). If omitted with --calibrate, a default fallback of 1.0 µm/pixel is used in case calibration fails.")
    parser.add_argument('--outdir', type=str, required=True, help="Output directory for results")
    parser.add_argument('--page', type=int, default=None, help="Page index for multi‑page TIFF (0‑based)")
    parser.add_argument('--qc', action='store_true', help="Save quality control panels")
    parser.add_argument('--recursive', action='store_true',
                        help="Recursively search for images within subdirectories when --input_dir is specified."
                             " If enabled, the relative folder structure is replicated under --outdir and recorded"
                             " in the output CSV.")
    parser.add_argument('--calibrate', action='store_true',
                        help="Interactively calibrate pixel size for each image by clicking on the scale bar."
                             " If specified, the provided --pixel_size_um value acts as a fallback if calibration fails.")
    parser.add_argument('--auto', action='store_true',
                        help="Use automatic region-growing annotation instead of manual polygon drawing.")
    parser.add_argument('--lo_diff', type=int, default=20,
                        help="Lower intensity tolerance for automatic flood-fill annotation (default: 20)")
    parser.add_argument('--hi_diff', type=int, default=20,
                        help="Upper intensity tolerance for automatic flood-fill annotation (default: 20)")
    parser.add_argument('--auto_polygon', action='store_true',
                        help="Use polygon ROI with Otsu-based automatic segmentation. Overrides --auto if both are set.")
    parser.add_argument('--polygon_method', type=str, default='otsu', choices=['otsu', 'grabcut'],
                        help="Segmentation method for polygon-based annotation ('otsu' or 'grabcut')."
                             " Default: 'otsu'.")
    parser.add_argument('--edge_mode', type=str, default='single', choices=['single', 'dualcurve'],
                        help="Edge mode for segmentation. 'single' uses standard segmentation. "
                             "'dualcurve' allows the user to define tip, root start, and root end to refine the boundary.")
    args = parser.parse_args()

    # Validate pixel size and calibration options.  If calibration is not requested
    # and no pixel size is provided, the user must supply a default pixel size.
    if not args.calibrate and args.pixel_size_um is None:
        parser.error("Either --pixel_size_um must be provided or --calibrate must be used to set pixel sizes interactively.")
    # Provide a fallback pixel size when calibrating if none is specified
    if args.calibrate and args.pixel_size_um is None:
        args.pixel_size_um = 1.0

    if args.image is None and args.input_dir is None:
        parser.error("Either --image or --input_dir must be specified")
    # Ensure the main output directory exists so that CSV can be written
    if args.outdir:
        try:
            os.makedirs(args.outdir, exist_ok=True)
        except Exception:
            # Defer errors to later when writing files
            pass
    # Collect images to process
    tasks: List[Tuple[str, Optional[str], str]] = []  # (image_path, mask_path, subdir)
    if args.image:
        # Single image; subdir is empty
        tasks.append((args.image, args.mask, ''))
    if args.input_dir:
        supported = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        # Keep track of processed file hashes to skip duplicates
        processed_hashes: set = set()
        if args.recursive:
            # Recursively walk through subdirectories
            for dirpath, dirnames, filenames in os.walk(args.input_dir):
                # Compute subdir relative to root
                rel_dir = os.path.relpath(dirpath, args.input_dir)
                # Normalize '.' to empty string for top level
                subdir = '' if rel_dir == '.' else rel_dir
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    base_no_ext = os.path.splitext(fname)[0]
                    # Skip mask files (those ending with _inst)
                    if base_no_ext.endswith('_inst'):
                        continue
                    if ext in supported:
                        img_path = os.path.join(dirpath, fname)
                        # Compute hash of the image to detect duplicates
                        try:
                            with open(img_path, 'rb') as fh:
                                file_bytes = fh.read()
                            md5 = hashlib.md5(file_bytes).hexdigest()
                        except Exception:
                            md5 = None
                        if md5 and md5 in processed_hashes:
                            # Skip duplicate image content
                            continue
                        if md5:
                            processed_hashes.add(md5)
                        base_name = base_no_ext
                        # Look for a mask with the same base name in the same folder
                        mask_path = None
                        for suffix in ['_inst.png', '_inst.tif', '_inst.tiff']:
                            candidate = os.path.join(dirpath, base_name + suffix)
                            if os.path.exists(candidate):
                                mask_path = candidate
                                break
                        tasks.append((img_path, mask_path, subdir))
        else:
            # Only process files in the top level of the directory
            for fname in os.listdir(args.input_dir):
                ext = os.path.splitext(fname)[1].lower()
                base_no_ext = os.path.splitext(fname)[0]
                # Skip mask files
                if base_no_ext.endswith('_inst'):
                    continue
                if ext in supported:
                    img_path = os.path.join(args.input_dir, fname)
                    # Compute hash to detect duplicate images
                    try:
                        with open(img_path, 'rb') as fh:
                            file_bytes = fh.read()
                        md5 = hashlib.md5(file_bytes).hexdigest()
                    except Exception:
                        md5 = None
                    if md5 and md5 in processed_hashes:
                        continue
                    if md5:
                        processed_hashes.add(md5)
                    base_name = base_no_ext
                    mask_path = None
                    for suffix in ['_inst.png', '_inst.tif', '_inst.tiff']:
                        candidate = os.path.join(args.input_dir, base_name + suffix)
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break
                    tasks.append((img_path, mask_path, ''))
    # Process each task
    all_features: List[Dict[str, float]] = []
    for img_path, mask_path, subdir in tasks:
        page = args.page
        feat_list = process_single_image(
            img_path,
            mask_path,
            args.pixel_size_um,
            args.outdir,
            page=page,
            qc=args.qc,
            subdir=subdir,
            calibrate=args.calibrate,
            auto_annotation=args.auto,
            lo_diff=args.lo_diff,
            hi_diff=args.hi_diff,
            auto_polygon=args.auto_polygon,
            polygon_method=args.polygon_method,
            edge_mode=args.edge_mode
        )
        all_features.extend(feat_list)
    # Save combined CSV
    import pandas as pd  # local import to avoid mandatory dependency
    df = pd.DataFrame(all_features)
    csv_path = os.path.join(args.outdir, 'features_summary.csv')
    # Attempt to write the combined CSV.  If the file is open or otherwise
    # locked (common on Windows when the CSV is open in Excel), a PermissionError
    # may be raised.  In that case, write to a timestamped filename instead.
    actual_csv_path = csv_path
    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        # If we cannot write due to a permission issue, fall back to a
        # timestamped filename to avoid failing entirely.  Notify the user.
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        alt_path = os.path.join(args.outdir, f'features_summary_{ts}.csv')
        print(f"Warning: Unable to write '{csv_path}' due to permission issues. "
              f"Saving to alternate file '{alt_path}' instead.")
        df.to_csv(alt_path, index=False)
        actual_csv_path = alt_path
    print(f"Saved features for {len(all_features)} instances to '{actual_csv_path}'")


# Delay execution of main() until all helper functions are defined.  The
# call to main() is moved to the very end of this file so that functions
# defined below (e.g. auto_segment helpers) are available when main() runs.
# When this script is executed directly (not imported), the following
# conditional at the bottom will invoke main().

# ----------------------------------------------------------------------
# Automatic segmentation helper functions and annotation modes
# ----------------------------------------------------------------------

def auto_segment_region(image: np.ndarray, seed: Tuple[int, int], lo_diff: int = 20, hi_diff: int = 20) -> np.ndarray:
    """
    Automatically segment a region around a seed point using flood fill.

    Converts the input image to grayscale if needed, applies OpenCV's floodFill
    starting from the given seed, and cleans the resulting mask with
    morphological opening and closing.

    Args:
        image: Input image array (grayscale or RGB).
        seed: (x, y) coordinates of the seed point in (column, row) order.
        lo_diff: Lower brightness tolerance for flood fill.
        hi_diff: Upper brightness tolerance for flood fill.

    Returns:
        A boolean mask (True where the region belongs to the object).
    """
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    x, y = seed
    h, w = gray.shape
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), dtype=bool)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    img_copy = gray.copy()
    cv2.floodFill(img_copy, flood_mask, (x, y), newVal=255, loDiff=lo_diff, upDiff=hi_diff, flags=4)
    region = flood_mask[1:-1, 1:-1].astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    region_u8 = region.astype(np.uint8)
    region_clean = cv2.morphologyEx(region_u8, cv2.MORPH_OPEN, kernel)
    region_clean = cv2.morphologyEx(region_clean, cv2.MORPH_CLOSE, kernel)
    return region_clean.astype(bool)


def auto_segment_in_polygon(image: np.ndarray, polygon_mask: np.ndarray) -> np.ndarray:
    """
    Automatically segment a trichome within a user-defined polygon ROI using Otsu thresholding.

    Args:
        image: Input image (grayscale or RGB).
        polygon_mask: Boolean array where True indicates the user-defined ROI.

    Returns:
        A boolean mask of the segmented trichome within the ROI.
    """
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    h, w = gray.shape
    if polygon_mask.shape != (h, w):
        return np.zeros((h, w), dtype=bool)
    region_pixels = gray[polygon_mask]
    if region_pixels.size == 0:
        return np.zeros((h, w), dtype=bool)
    # Convert to uint8 for thresholding
    if region_pixels.dtype != np.uint8:
        max_val = float(region_pixels.max()) if region_pixels.max() > 0 else 1.0
        region_pixels_uint8 = (region_pixels / max_val * 255).astype(np.uint8)
    else:
        region_pixels_uint8 = region_pixels.copy()
    ret_val, _ = cv2.threshold(region_pixels_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = np.zeros_like(gray, dtype=np.uint8)
    if region_pixels.max() > 0:
        if region_pixels.dtype != np.uint8:
            threshold_actual = (ret_val / 255.0) * float(region_pixels.max())
        else:
            threshold_actual = ret_val
        binary_mask[polygon_mask] = (gray[polygon_mask] >= threshold_actual).astype(np.uint8) * 255
    else:
        return np.zeros((h, w), dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return (binary_mask > 0)


def auto_segment_in_polygon_grabcut(image: np.ndarray, polygon_mask: np.ndarray) -> np.ndarray:
    """
    Automatically segment a trichome within a user-defined polygon ROI using
    the GrabCut algorithm.  The polygon region is treated as probable
    foreground while everything outside is probable background.  After
    segmentation, morphological operations smooth and clean the mask.

    Args:
        image: Input image (grayscale or RGB).
        polygon_mask: Boolean array where True indicates the user-defined ROI.

    Returns:
        A boolean mask of the segmented trichome within the ROI.
    """
    # Convert image to BGR if needed (GrabCut expects 3-channel)
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] >= 3:
        # Ensure BGR ordering for OpenCV
        if image.shape[2] == 3:
            # Assume input is grayscale or BGR.  If input is RGB, swap channels.
            # We'll heuristically detect if the first channel has more contrast
            # to decide if it's likely RGB.  In ambiguous cases, use original.
            # For simplicity, assume image is BGR as loaded by cv2.
            img_bgr = image.copy()
        else:
            img_bgr = image[:, :, :3].copy()
    else:
        # Unknown format
        return np.zeros_like(polygon_mask, dtype=bool)
    h, w = polygon_mask.shape
    if polygon_mask.shape != (h, w):
        return np.zeros((h, w), dtype=bool)
    # Create initial mask for GrabCut
    # 0: background, 1: foreground, 2: probable background, 3: probable foreground
    grab_mask = np.zeros((h, w), dtype=np.uint8)
    # outside polygon is definite background
    grab_mask[~polygon_mask] = cv2.GC_BGD
    # inside polygon is probable foreground
    grab_mask[polygon_mask] = cv2.GC_PR_FGD
    # Setup background and foreground models (GrabCut uses these arrays internally)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, grab_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        # GrabCut can fail on empty or uniform images
        return np.zeros((h, w), dtype=bool)
    # Foreground if mask pixel is GC_FGD (1) or GC_PR_FGD (3)
    mask_fg = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    # Morphological cleaning to remove noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Opening removes small false positives
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing fills small gaps and smooths edges
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Optional: small edge snap using Canny edges to refine borders
    # Compute edges on grayscale
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    edges = cv2.Canny(gray, 50, 150)
    # Dilate edges slightly so that boundary pixels can snap to them
    edge_dil = cv2.dilate(edges, kernel, iterations=1)
    # For each pixel in the mask that is adjacent to background, if an edge is present, keep it
    # Compute mask boundary
    boundary = cv2.morphologyEx(mask_fg, cv2.MORPH_GRADIENT, kernel)
    snap = np.logical_and(boundary.astype(bool), edge_dil.astype(bool))
    # Add snapped pixels to mask
    mask_fg[snap] = 1
    return (mask_fg > 0)


def refine_segmentation_grabcut(image: np.ndarray, current_seg: np.ndarray) -> np.ndarray:
    """
    Refine an existing segmentation mask using GrabCut.

    The current segmentation is treated as definite foreground, pixels in a
    small border region around it are treated as probable foreground, and
    everything outside is treated as background.  GrabCut then refines the
    foreground boundary.  The resulting mask is cleaned with morphological
    operations and edge snapping similar to the initial GrabCut pass.

    Args:
        image: Input image (grayscale or RGB).
        current_seg: Boolean array representing the existing segmentation.

    Returns:
        A boolean mask representing the refined segmentation.
    """
    if current_seg is None or not np.any(current_seg):
        return np.zeros_like(current_seg, dtype=bool)
    # Convert image to BGR for GrabCut
    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] >= 3:
        # Copy first 3 channels in case of alpha
        img_bgr = image[:, :, :3].copy()
    else:
        # unsupported format
        return current_seg.copy()
    h, w = current_seg.shape
    # Create GrabCut mask
    grab_mask = np.zeros((h, w), dtype=np.uint8)
    # Definite foreground where current segmentation is true
    grab_mask[current_seg] = cv2.GC_FGD
    # Determine a border region around the segmentation to mark as probable foreground
    # Compute bounding box of segmentation
    ys, xs = np.where(current_seg)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    margin = int(0.05 * max(h, w))  # 5% of image dimension
    y0 = max(0, y_min - margin)
    y1 = min(h, y_max + margin)
    x0 = max(0, x_min - margin)
    x1 = min(w, x_max + margin)
    # Mark pixels within bounding box but not in current segmentation as probable foreground
    grab_mask[y0:y1, x0:x1][~current_seg[y0:y1, x0:x1]] = cv2.GC_PR_FGD
    # Everything outside the bounding box is background
    # Run GrabCut
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, grab_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        # if GrabCut fails, return original segmentation
        return current_seg.copy()
    # Extract refined foreground
    refined = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Edge snap similar to auto_segment_in_polygon_grabcut
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    edges = cv2.Canny(gray, 50, 150)
    edge_dil = cv2.dilate(edges, kernel, iterations=1)
    boundary = cv2.morphologyEx(refined, cv2.MORPH_GRADIENT, kernel)
    snap = np.logical_and(boundary.astype(bool), edge_dil.astype(bool))
    refined[snap] = 1
    return (refined > 0)


def refine_segmentation_otsu(image: np.ndarray, current_seg: np.ndarray) -> np.ndarray:
    """
    Refine an existing Otsu segmentation mask by locally re‑thresholding around
    the existing mask and snapping to nearby edges.  This routine expands
    the current mask slightly, applies Otsu thresholding within that region,
    fills holes, and snaps the boundary to strong edges.  It is intended
    to improve segmentation obtained via the 'otsu' polygon method when
    boundaries are not well captured.

    Args:
        image: Input image (grayscale or RGB).
        current_seg: Boolean array representing the existing segmentation.

    Returns:
        A boolean mask representing the refined segmentation.
    """
    if current_seg is None or not np.any(current_seg):
        return np.zeros_like(current_seg, dtype=bool)
    # Convert image to grayscale
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    h, w = current_seg.shape
    # Dilate the current segmentation slightly to include nearby pixels
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(current_seg.astype(np.uint8), dil_kernel, iterations=1)
    # Determine bounding box of dilated mask
    ys, xs = np.where(dilated > 0)
    if len(xs) == 0:
        return current_seg.copy()
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    # Add margin around bounding box to capture context
    margin = int(0.05 * max(h, w))
    y0 = max(0, y_min - margin)
    y1 = min(h, y_max + margin)
    x0 = max(0, x_min - margin)
    x1 = min(w, x_max + margin)
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[y0:y1, x0:x1] = True
    # Extract pixel intensities within region
    region_pixels = gray[region_mask]
    if region_pixels.size == 0:
        return current_seg.copy()
    # Convert to uint8 for thresholding if needed
    if region_pixels.dtype != np.uint8:
        max_val = float(region_pixels.max()) if region_pixels.max() > 0 else 1.0
        region_uint8 = (region_pixels / max_val * 255).astype(np.uint8)
    else:
        region_uint8 = region_pixels.copy()
    ret_val, _ = cv2.threshold(region_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert threshold back to original scale if needed
    if region_pixels.dtype != np.uint8:
        threshold_actual = (ret_val / 255.0) * float(region_pixels.max())
    else:
        threshold_actual = ret_val
    new_mask = np.zeros_like(current_seg, dtype=np.uint8)
    # Apply threshold within region
    new_mask[region_mask] = (gray[region_mask] >= threshold_actual).astype(np.uint8)
    # Morphological opening and closing to clean noise
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, small_kernel, iterations=1)
    # Combine with current segmentation
    combined = np.logical_or(current_seg, new_mask > 0)
    # Fill holes within the combined mask
    try:
        filled = ndi.binary_fill_holes(combined).astype(bool)
    except Exception:
        filled = combined.copy()
    # Smooth the filled mask with closing
    filled_u8 = filled.astype(np.uint8)
    filled_u8 = cv2.morphologyEx(filled_u8, cv2.MORPH_CLOSE, small_kernel, iterations=1)
    # Edge snapping: snap boundary pixels to nearby strong edges
    edges = cv2.Canny(gray, 50, 150)
    edge_dil = cv2.dilate(edges, small_kernel, iterations=1)
    boundary = cv2.morphologyEx(filled_u8, cv2.MORPH_GRADIENT, small_kernel)
    snap = np.logical_and(boundary.astype(bool), edge_dil.astype(bool))
    filled_u8[snap] = 1
    return (filled_u8 > 0)


# ----------------------------------------------------------------------
# Dualcurve annotation for splitting contour into upper/lower/root paths
# ----------------------------------------------------------------------
def annotate_instance_dualcurve(image: np.ndarray, current_seg: np.ndarray):
    """
    Interactive tool to refine a single trichome segmentation by manually splitting
    its outer contour into tip‑to‑root "upper" and "lower" boundaries using three
    user‑selected points.  The user clicks on the tip, the root start, and the
    root end along the displayed segmentation boundary.  The region between the
    resulting upper and lower paths is filled to form a new segmentation mask.

    Args:
        image: The original image (grayscale or RGB) from which the segmentation
            was derived.  Used for display; the mask is not modified in
            content but only geometrically.
        current_seg: A boolean array representing the current segmentation of
            a single trichome instance (True for foreground).

    Returns:
        Tuple[np.ndarray, Dict[str, float]]: A tuple containing the (unmodified)
        segmentation mask and a dictionary of curvature metrics for the upper
        and lower boundaries.  If the user cancels or does not accept the
        refined segmentation, the original mask and NaN metrics are returned.
    """
    import matplotlib.pyplot as plt
    import math
    # Prepare display image in RGB for overlay
    if image.ndim == 2:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # Convert BGR to RGB if necessary
        if image.shape[2] >= 3:
            b, g, r = cv2.split(image[:, :, :3])
            display_img = cv2.merge([r, g, b])
        else:
            display_img = image[:, :, :3].copy()
    # Ensure current_seg is boolean
    seg_mask = current_seg.astype(bool)
    # Disable matplotlib's default save key ('s') so it doesn't override our custom swap
    import matplotlib as mpl
    mpl.rcParams['keymap.save'] = []
    # Initialize state
    click_points: List[Tuple[int, int]] = []  # stores raw click points (not snapped)
    refined_seg: Optional[np.ndarray] = None
    result_seg: np.ndarray = seg_mask.copy()
    # Variables to hold current upper, lower, and root paths after splitting.  These
    # will be used to allow swapping and to show previews.
    upper_path: Optional[List[Tuple[int, int]]] = None
    lower_path: Optional[List[Tuple[int, int]]] = None
    root_path_vars: Optional[List[Tuple[int, int]]] = None
    # Metrics for curvature of upper and lower boundary.  These will be returned
    # to the caller to be added to the feature table.  Initialise as NaNs.
    boundary_metrics: Dict[str, float] = {
        'upper_curv_mean_rad': float('nan'),
        'upper_curv_max_rad': float('nan'),
        'upper_curv_sum_rad': float('nan'),
        'lower_curv_mean_rad': float('nan'),
        'lower_curv_max_rad': float('nan'),
        'lower_curv_sum_rad': float('nan')
    }

    # Extract the outer contour of the segmentation for splitting
    seg_u8 = (seg_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return seg_mask.copy()
    # Choose the largest contour by area
    max_contour = max(contours, key=cv2.contourArea)
    # Flatten contour to list of (x, y) points (in image coordinate order)
    contour_points = [(int(pt[0][0]), int(pt[0][1])) for pt in max_contour]

    # Precompute an edge map for click snapping.  Convert to grayscale if needed.
    if image.ndim == 2:
        gray_for_edges = image.copy()
    else:
        # convert BGR or RGB to grayscale consistently
        if image.shape[2] >= 3:
            # If BGR, convert; if already RGB, the order doesn't matter for edge detection
            gray_for_edges = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray_for_edges = image[:, :, 0].copy()
    # Apply a moderate blur to suppress noise before Canny
    blurred_edges = cv2.GaussianBlur(gray_for_edges, (5, 5), 0)
    edges = cv2.Canny(blurred_edges, 50, 150)

    # Utility to snap a click to the nearest strong edge within a small radius.  This helps
    # the user select precise contour points even if the mouse click is slightly off.
    def snap_to_edge(pt: Tuple[int, int], edge_img: np.ndarray, radius: int = 8) -> Tuple[int, int]:
        x, y = pt
        h, w = edge_img.shape
        # Define a neighbourhood around the click
        x0, x1 = max(0, x - radius), min(w - 1, x + radius)
        y0, y1 = max(0, y - radius), min(h - 1, y + radius)
        roi = edge_img[y0:y1 + 1, x0:x1 + 1]
        # If no edges are present nearby, return original point
        if roi is None or roi.size == 0 or not np.any(roi):
            return (x, y)
        # Coordinates of edge pixels within ROI
        ys, xs = np.where(roi > 0)
        # Convert local coordinates back to image coordinates
        xs = xs + x0
        ys = ys + y0
        # Find the edge pixel closest to the click location
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        idx = int(np.argmin(d2))
        return (int(xs[idx]), int(ys[idx]))

    # Compute curvature statistics for a polyline path.  Curvature is approximated
    # via the turning angle between successive segments.  Absolute curvature
    # values are used for summarising.
    def compute_arc_curvatures(path: List[Tuple[int, int]]) -> Tuple[float, float, float]:
        """Compute mean, max and sum of absolute curvature along a polyline.

        Args:
            path: List of (x, y) points representing the polyline.

        Returns:
            (mean_abs_curv, max_abs_curv, sum_abs_curv) in radians.  If the path
            has fewer than 3 points, all values are NaN.
        """
        if path is None or len(path) < 3:
            return float('nan'), float('nan'), float('nan')
        curv_vals: List[float] = []
        for i in range(1, len(path) - 1):
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            # Vectors from point i
            v1x = x1 - x0
            v1y = y1 - y0
            v2x = x2 - x1
            v2y = y2 - y1
            # Compute angle between v1 and v2 via cross/dot
            # Protect against zero-length vectors
            len1 = math.hypot(v1x, v1y)
            len2 = math.hypot(v2x, v2y)
            if len1 == 0 or len2 == 0:
                continue
            dot = v1x * v2x + v1y * v2y
            cross = v1x * v2y - v1y * v2x
            # angle between vectors
            angle = math.atan2(cross, dot)
            curv_vals.append(angle)
        if not curv_vals:
            return float('nan'), float('nan'), float('nan')
        # Use absolute curvature values
        abs_vals = [abs(c) for c in curv_vals]
        mean_c = float(np.mean(abs_vals))
        max_c = float(np.max(abs_vals))
        sum_c = float(np.sum(abs_vals))
        return mean_c, max_c, sum_c

    # Local helper to compute the closest index on the contour to a given point
    def closest_index_on_poly(pt: Tuple[int, int], poly_xy: List[Tuple[int, int]]) -> int:
        px, py = pt
        min_dist = None
        min_idx = 0
        for i, (cx, cy) in enumerate(poly_xy):
            dx = cx - px
            dy = cy - py
            d2 = dx * dx + dy * dy
            if (min_dist is None) or (d2 < min_dist):
                min_dist = d2
                min_idx = i
        return min_idx

    # Extract a subpath along the polygon from i to j (inclusive), wrapping if needed
    def subpath(poly: List[Tuple[int, int]], i: int, j: int) -> List[Tuple[int, int]]:
        if i <= j:
            return poly[i:j + 1]
        return poly[i:] + poly[:j + 1]

    # Classify which of two candidate arcs is the visually upper versus lower arc using PCA.
    # We rotate the points into the principal component basis and compare the mean of the
    # second principal coordinate.  In the rotated frame, a larger y coordinate means the
    # path is visually lower (because screen y increases downwards).  We then assign
    # accordingly.  If the classification is inverted, the user can press 's' to swap later.
    def classify_upper_lower(path_a: List[Tuple[int, int]], path_b: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        import numpy as _np  # local import to avoid name clashes
        pts = _np.array(path_a + path_b, dtype=float)
        if pts.shape[0] < 3:
            # Not enough points to perform PCA; fallback to average y comparison
            avg_y_a = _np.mean([p[1] for p in path_a]) if path_a else 0
            avg_y_b = _np.mean([p[1] for p in path_b]) if path_b else 0
            if avg_y_a > avg_y_b:
                return path_b, path_a
            else:
                return path_a, path_b
        # Center the points
        mu = pts.mean(axis=0)
        X = pts - mu
        # Compute SVD for PCA
        U, s, Vt = _np.linalg.svd(X, full_matrices=False)
        R = Vt  # rows of Vt are principal axes
        # Project each path onto rotated coordinates
        def rot_y(path: List[Tuple[int, int]]) -> float:
            P = _np.array(path, dtype=float) - mu
            Pr = P @ R.T
            return _np.mean(Pr[:, 1]) if Pr.size > 0 else 0.0
        y_a = rot_y(path_a)
        y_b = rot_y(path_b)
        # In image coordinates, larger y means lower (toward bottom).  In rotated frame,
        # we treat the second component similarly.  So whichever has larger mean y is lower.
        if y_a > y_b:
            return path_b, path_a  # path_a is lower; swap
        else:
            return path_a, path_b

    def split_contour_by_points(contour_xy: List[Tuple[int, int]], tip_pt: Tuple[int, int], root_start_pt: Tuple[int, int], root_end_pt: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Split the contour into upper, lower, and root paths given three points.

        Three arcs are formed between consecutive clicked points along the contour.
        The arc with the highest average y (visually lowest) is designated as the root path.
        The remaining two arcs are classified into upper and lower boundaries using PCA.

        Args:
            contour_xy: List of contour (x, y) tuples in traversal order.
            tip_pt: Coordinates of the tip point selected by the user.
            root_start_pt: Coordinates of the root start point.
            root_end_pt: Coordinates of the root end point.

        Returns:
            upper_path: List of points representing the upper boundary.
            lower_path: List of points representing the lower boundary.
            root_path: List of points representing the root boundary.
        """
        # Snap each click to the nearest edge point for higher accuracy
        tip_pt_snapped = snap_to_edge(tip_pt, edges)
        root_start_snapped = snap_to_edge(root_start_pt, edges)
        root_end_snapped = snap_to_edge(root_end_pt, edges)
        # Find indices on the contour for each snapped point
        i_tip = closest_index_on_poly(tip_pt_snapped, contour_xy)
        i_rs = closest_index_on_poly(root_start_snapped, contour_xy)
        i_re = closest_index_on_poly(root_end_snapped, contour_xy)
        # Build all three arcs defined by the three points.  The contour is assumed to be
        # closed, so arcs wrap around as needed.
        arc_tip_to_rs = subpath(contour_xy, i_tip, i_rs)
        arc_rs_to_re = subpath(contour_xy, i_rs, i_re)
        arc_re_to_tip = subpath(contour_xy, i_re, i_tip)
        # Determine which arc is the root path: the arc with the highest mean y (lowest on screen)
        arcs = [arc_tip_to_rs, arc_rs_to_re, arc_re_to_tip]
        avg_ys = []
        for arc in arcs:
            # If arc has no points, assign a very low y to avoid selecting it as root
            if len(arc) == 0:
                avg_ys.append(-float('inf'))
            else:
                avg_ys.append(float(np.mean([p[1] for p in arc])))
        # Index of the arc with the maximum average y coordinate
        root_idx = int(np.argmax(avg_ys))
        root_path = arcs[root_idx]
        # The remaining two arcs are candidates for upper/lower boundaries
        cand_paths = [arcs[(root_idx + 1) % 3], arcs[(root_idx + 2) % 3]]
        # Classify the remaining two arcs into upper and lower using PCA-based method
        upper_path, lower_path = classify_upper_lower(cand_paths[0], cand_paths[1])
        return upper_path, lower_path, root_path

    # Legacy helper functions from the previous implementation.  They are kept
    # under unused names to avoid overriding the newer versions defined earlier.
    def _legacy_find_closest_contour_index(pt: Tuple[int, int], contour_xy: List[Tuple[int, int]]) -> int:
        px, py = pt
        # Compute squared distances to all contour points
        min_dist = None
        min_idx = 0
        for i, (cx, cy) in enumerate(contour_xy):
            dx = cx - px
            dy = cy - py
            d2 = dx * dx + dy * dy
            if (min_dist is None) or (d2 < min_dist):
                min_dist = d2
                min_idx = i
        return min_idx

    def _legacy_get_path(contour_xy: List[Tuple[int, int]], start_idx: int, end_idx: int) -> List[Tuple[int, int]]:
        # Return the path along contour from start_idx to end_idx (inclusive), wrapping if needed
        if start_idx <= end_idx:
            return contour_xy[start_idx:end_idx + 1]
        else:
            return contour_xy[start_idx:] + contour_xy[:end_idx + 1]

    def _legacy_split_contour_by_points(contour_xy: List[Tuple[int, int]], tip_pt: Tuple[int, int], root_start_pt: Tuple[int, int], root_end_pt: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Unused legacy split function retained for backward reference."""
        i_tip = _legacy_find_closest_contour_index(tip_pt, contour_xy)
        i_rs = _legacy_find_closest_contour_index(root_start_pt, contour_xy)
        i_re = _legacy_find_closest_contour_index(root_end_pt, contour_xy)
        # Paths along contour
        path1 = _legacy_get_path(contour_xy, i_tip, i_rs)
        path2 = _legacy_get_path(contour_xy, i_re, i_tip)
        root_path = _legacy_get_path(contour_xy, i_rs, i_re)
        avg_y1 = np.mean([p[1] for p in path1]) if path1 else 0
        avg_y2 = np.mean([p[1] for p in path2]) if path2 else 0
        if avg_y1 > avg_y2:
            upper_path = path1
            lower_path = path2
        else:
            upper_path = path2
            lower_path = path1
        return upper_path, lower_path, root_path

    # Figure for interactive selection
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title('Select tip, root start, root end (left click). Keys: y=accept, r=reset, q=cancel')
    ax.imshow(display_img)
    ax.axis('off')

    # Overlay helper function.  Displays the base image, current segmentation (or refined
    # segmentation), and any clicked points.  Optionally draws the upper/lower/root
    # contours when available.
    def update_overlay_dual(show_paths: bool = False):
        ax.clear()
        # Base image
        ax.imshow(display_img)
        # Always draw the original segmentation (result_seg) as a semi-transparent red mask
        base_mask = result_seg
        if base_mask is not None and np.any(base_mask):
            overlay = display_img.astype(float).copy()
            nm = base_mask.astype(float)
            overlay[:, :, 0] = overlay[:, :, 0] * (1 - 0.5 * nm) + 255 * (0.5 * nm)
            overlay[:, :, 1] = overlay[:, :, 1] * (1 - 0.5 * nm)
            overlay[:, :, 2] = overlay[:, :, 2] * (1 - 0.5 * nm)
            ax.imshow(overlay.astype(np.uint8))
        # If asked, draw the current upper, lower, and root paths
        if show_paths and upper_path is not None and lower_path is not None and root_path_vars is not None:
            # draw upper (red), lower (blue), root (green)
            ax.plot([p[0] for p in upper_path], [p[1] for p in upper_path], color='red', linewidth=1)
            ax.plot([p[0] for p in lower_path], [p[1] for p in lower_path], color='blue', linewidth=1)
            ax.plot([p[0] for p in root_path_vars], [p[1] for p in root_path_vars], color='green', linewidth=1)
        # Draw click points with labels
        colors_pts = ['red', 'lime', 'magenta']
        labels = ['tip', 'root start', 'root end']
        for i, pt in enumerate(click_points):
            ax.plot(pt[0], pt[1], marker='o', color=colors_pts[i], markersize=5)
            ax.text(pt[0] + 2, pt[1] + 2, labels[i], color=colors_pts[i], fontsize=8)
        ax.axis('off')
        fig.canvas.draw()

    # Event handlers
    def on_click(event):
        nonlocal refined_seg, result_seg, upper_path, lower_path, root_path_vars
        if event.inaxes != ax:
            return
        # Only handle left clicks
        if event.button != 1:
            return
        # Require valid coordinates
        if event.xdata is None or event.ydata is None:
            return
        # Record the raw click point
        x_click = int(round(event.xdata))
        y_click = int(round(event.ydata))
        click_points.append((x_click, y_click))
        update_overlay_dual()
        # When three points are selected, split contour and build the refined mask
        if len(click_points) == 3:
            # Retrieve points
            tip_pt, rs_pt, re_pt = click_points
            # Compute paths using snapped points and classification
            up, low, rootp = split_contour_by_points(contour_points, tip_pt, rs_pt, re_pt)
            # Save the paths for possible swapping
            upper_path = up
            lower_path = low
            root_path_vars = rootp
            # Compute curvature metrics for the current upper and lower paths
            u_mean, u_max, u_sum = compute_arc_curvatures(upper_path)
            l_mean, l_max, l_sum = compute_arc_curvatures(lower_path)
            boundary_metrics['upper_curv_mean_rad'] = u_mean
            boundary_metrics['upper_curv_max_rad'] = u_max
            boundary_metrics['upper_curv_sum_rad'] = u_sum
            boundary_metrics['lower_curv_mean_rad'] = l_mean
            boundary_metrics['lower_curv_max_rad'] = l_max
            boundary_metrics['lower_curv_sum_rad'] = l_sum
            # Build polygon from upper and reversed lower path
            poly_pts = np.array(upper_path + list(reversed(lower_path)), dtype=np.int32)
            new_mask = np.zeros_like(seg_mask, dtype=np.uint8)
            if poly_pts.size > 0:
                cv2.fillPoly(new_mask, [poly_pts], 1)
            # Morphological closing to smooth small gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            # Assign to refined_seg for preview (not used to modify result_seg)
            refined_seg = (new_mask > 0)
            # Update overlay and show the paths
            update_overlay_dual(show_paths=True)

    def on_key(event):
        nonlocal refined_seg, result_seg, click_points, upper_path, lower_path, root_path_vars
        # Normalize the key to lowercase string, handle None gracefully
        k = ''
        if event.key is not None:
            try:
                k = event.key.lower()
            except Exception:
                k = str(event.key).lower()
        # Accept: do not modify the segmentation mask; just close the figure
        if k == 'y':
            # On acceptance, we keep the original mask unchanged.  We do not assign
            # refined_seg to result_seg so the instance segmentation remains
            # identical to current_seg.
            plt.close(fig)
        # Reset current selection: clear points and temporary refinements
        elif k == 'r':
            click_points.clear()
            refined_seg = None
            upper_path = None
            lower_path = None
            root_path_vars = None
            update_overlay_dual()
        # Cancel: close figure without modifying segmentation
        elif k == 'q':
            refined_seg = None
            plt.close(fig)
        # Swap: exchange the upper and lower paths (if defined) and update the overlay.
        elif k == 's':
            if upper_path is not None and lower_path is not None:
                # Swap upper and lower paths
                upper_path, lower_path = lower_path, upper_path
                # Recompute curvature metrics for swapped paths
                u_mean, u_max, u_sum = compute_arc_curvatures(upper_path)
                l_mean, l_max, l_sum = compute_arc_curvatures(lower_path)
                boundary_metrics['upper_curv_mean_rad'] = u_mean
                boundary_metrics['upper_curv_max_rad'] = u_max
                boundary_metrics['upper_curv_sum_rad'] = u_sum
                boundary_metrics['lower_curv_mean_rad'] = l_mean
                boundary_metrics['lower_curv_max_rad'] = l_max
                boundary_metrics['lower_curv_sum_rad'] = l_sum
                # Recreate a preview mask for overlay (mask is not used to modify result_seg)
                poly_pts = np.array(upper_path + list(reversed(lower_path)), dtype=np.int32)
                tmp_mask = np.zeros_like(seg_mask, dtype=np.uint8)
                if poly_pts.size > 0:
                    cv2.fillPoly(tmp_mask, [poly_pts], 1)
                # Smooth mask edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                tmp_mask = cv2.morphologyEx(tmp_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                refined_seg = (tmp_mask > 0)
                update_overlay_dual(show_paths=True)

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    update_overlay_dual()
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    return result_seg, boundary_metrics


def annotate_image_auto(image: np.ndarray, lo_diff: int = 20, hi_diff: int = 20) -> Tuple[np.ndarray, bool]:
    """
    Interactively annotate trichome regions via automatic segmentation from a seed.

    Users click inside a trichome to seed a flood-fill segmenter.  The detected
    region is shown as a red overlay.  Keys: 'y' to accept, 'r' to reject,
    'q' to finish, 'x' to skip.

    Args:
        image: Input image (grayscale or RGB).
        lo_diff: Lower intensity tolerance for flood fill.
        hi_diff: Upper intensity tolerance for flood fill.

    Returns:
        mask: Integer label mask.
        skipped: True if the user skipped the image.
    """
    if image.ndim == 2:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        if image.shape[2] == 3:
            b, g, r = cv2.split(image)
            display_img = cv2.merge([r, g, b])
        else:
            display_img = image[:, :, :3].copy()
    mask_result = np.zeros(image.shape[:2], dtype=np.int32)
    current_seg: Optional[np.ndarray] = None
    inst_id = 1
    skip_image = False
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display_img)
    ax.set_title('Auto annotate: click to segment, y=accept, r=reject, q=finish, x=skip')
    ax.axis('off')
    def update_overlay():
        ax.clear()
        ax.imshow(display_img)
        if current_seg is not None and np.any(current_seg):
            overlay = display_img.astype(float).copy()
            nm = current_seg.astype(float)
            overlay[:, :, 0] = overlay[:, :, 0] * (1 - 0.5 * nm) + 255 * (0.5 * nm)
            overlay[:, :, 1] = overlay[:, :, 1] * (1 - 0.5 * nm)
            overlay[:, :, 2] = overlay[:, :, 2] * (1 - 0.5 * nm)
            ax.imshow(overlay.astype(np.uint8))
        ax.set_title('Auto annotate: click to segment, y=accept, r=reject, q=finish, x=skip')
        ax.axis('off')
        fig.canvas.draw()
    def on_click(event):
        nonlocal current_seg
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        cx, cy = int(event.xdata + 0.5), int(event.ydata + 0.5)
        seg = auto_segment_region(image, (cx, cy), lo_diff=lo_diff, hi_diff=hi_diff)
        if np.any(seg):
            current_seg = seg
            update_overlay()
    def on_key(event):
        nonlocal current_seg, inst_id, skip_image
        if event.key == 'y':
            if current_seg is not None and np.any(current_seg):
                mask_result[current_seg] = inst_id
                inst_id += 1
                current_seg = None
                update_overlay()
        elif event.key == 'r':
            current_seg = None
            update_overlay()
        elif event.key == 'q':
            plt.close(fig)
        elif event.key == 'x':
            skip_image = True
            plt.close(fig)
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    if skip_image:
        return None, True
    return mask_result, False


def annotate_image_auto_polygon(image: np.ndarray, method: str = 'otsu', edge_mode: str = 'single') -> Tuple[np.ndarray, bool]:
    """
    Interactively annotate trichomes by drawing a polygon ROI and auto-segmenting inside.

    The user draws a polygon (left click to add vertices, right click to close) around a trichome.
    Depending on the selected method, segmentation within the polygon is performed
    using either Otsu thresholding ('otsu') or GrabCut with edge snapping ('grabcut').
    Keys: 'y' to accept, 'r' to reject and redraw, 'q' to finish, 'x' to skip.

    Args:
        image: Input image (grayscale or RGB).
        method: Segmentation method to use inside the polygon ('otsu' or 'grabcut').

    Returns:
        mask: Integer label mask of segmented instances.
        skipped: True if the user skipped annotation for this image.
    """
    # Prepare display image in RGB order for matplotlib
    if image.ndim == 2:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        if image.shape[2] == 3:
            # Convert BGR to RGB for display
            b, g, r = cv2.split(image)
            display_img = cv2.merge([r, g, b])
        else:
            display_img = image[:, :, :3].copy()
    mask_result = np.zeros(image.shape[:2], dtype=np.int32)
    inst_id = 1
    skip_image = False
    current_polygon: List[Tuple[float, float]] = []
    current_seg: Optional[np.ndarray] = None
    paint_mode: Optional[str] = None  # None, 'paint', or 'erase'
    # Refine mode: when True, the user draws a new polygon to refine the current segmentation.
    # In refine mode, automatic segmentation is performed inside the drawn polygon and the
    # resulting mask is combined with the existing current_seg (union).  This allows the
    # user to refine the segmentation without expanding it outside manually drawn regions.
    refine_mode: bool = False
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display_img)
    # Set initial title; will be updated dynamically in update_overlay based on refine and edge mode
    ax.set_title('')
    ax.axis('off')

    def update_overlay():
        # Redraw base image
        ax.clear()
        ax.imshow(display_img)
        # Draw current polygon lines
        if current_polygon:
            poly_arr = np.array(current_polygon)
            # Use different colours for paint/erase vs segmentation polygons
            if paint_mode == 'paint':
                col = 'c'  # cyan for painting
            elif paint_mode == 'erase':
                col = 'm'  # magenta for erasing
            else:
                col = 'y'  # yellow for segmentation
            ax.plot(poly_arr[:, 0], poly_arr[:, 1], col + '.-')
            if len(poly_arr) > 2:
                # Close the polygon visually
                ax.plot([poly_arr[-1, 0], poly_arr[0, 0]], [poly_arr[-1, 1], poly_arr[0, 1]], col + '-')
        # Draw current segmentation overlay if present
        if current_seg is not None and np.any(current_seg):
            overlay = display_img.astype(float).copy()
            nm = current_seg.astype(float)
            # Red tint for segmentation
            overlay[:, :, 0] = overlay[:, :, 0] * (1 - 0.5 * nm) + 255 * (0.5 * nm)
            overlay[:, :, 1] = overlay[:, :, 1] * (1 - 0.5 * nm)
            overlay[:, :, 2] = overlay[:, :, 2] * (1 - 0.5 * nm)
            ax.imshow(overlay.astype(np.uint8))
        # Update the title to reflect refine mode status
        if refine_mode:
            title = ('Refine mode: left=add, right=close to union with current, u=exit refine, '
                     'i=paint, e=erase, h=fill holes, y=accept, r=reject, q=finish, x=skip')
        else:
            if edge_mode == 'dualcurve':
                title = ('Auto polygon (dualcurve): left=add, right=close, i=paint, e=erase, h=fill holes, u=refine, '
                         'y=accept (define tip/root), r=reject, q=finish, x=skip')
            else:
                title = ('Auto polygon: left=add, right=close, i=paint, e=erase, h=fill holes, u=refine, '
                         'y=accept, r=reject, q=finish, x=skip')
        ax.set_title(title)
        ax.axis('off')
        fig.canvas.draw()

    def on_click(event):
        nonlocal current_polygon, current_seg, paint_mode
        if event.inaxes != ax:
            return
        # Collect polygon vertices for either segmentation or painting
        if event.button == 1:  # left click
            # add vertex to polygon
            if event.xdata is None or event.ydata is None:
                return
            current_polygon.append((event.xdata, event.ydata))
            update_overlay()
        elif event.button == 3:  # right click closes polygon
            if len(current_polygon) >= 3:
                poly_pts = np.array([(int(x + 0.5), int(y + 0.5)) for x, y in current_polygon], dtype=np.int32)
                # Create a mask for the polygon
                poly_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(poly_mask, [poly_pts], 1)
                poly_bool = poly_mask.astype(bool)
                if paint_mode:
                    # Paint or erase on current segmentation
                    if current_seg is None or not np.any(current_seg):
                        current_seg = np.zeros(image.shape[:2], dtype=bool)
                    if paint_mode == 'paint':
                        current_seg = np.logical_or(current_seg, poly_bool)
                    elif paint_mode == 'erase':
                        current_seg = np.logical_and(current_seg, np.logical_not(poly_bool))
                    # Exit paint mode
                    paint_mode = None
                    current_polygon = []
                    update_overlay()
                elif refine_mode:
                    # Refine mode: auto segment inside polygon and union with current segmentation
                    if method == 'grabcut':
                        seg = auto_segment_in_polygon_grabcut(image, poly_bool)
                    else:
                        seg = auto_segment_in_polygon(image, poly_bool)
                    if current_seg is None or not np.any(current_seg):
                        current_seg = seg.astype(bool)
                    else:
                        current_seg = np.logical_or(current_seg, seg)
                    current_polygon = []
                    update_overlay()
                else:
                    # Normal segmentation inside ROI
                    if method == 'grabcut':
                        seg = auto_segment_in_polygon_grabcut(image, poly_bool)
                    else:
                        seg = auto_segment_in_polygon(image, poly_bool)
                    if np.any(seg):
                        current_seg = seg.astype(bool)
                    current_polygon = []
                    update_overlay()

    def on_key(event):
        # Declare nonlocal variables that will be modified within this handler.  In particular,
        # refine_mode must be declared nonlocal so that toggling it via the 'u' key actually
        # updates the refine_mode flag in the outer scope.  Without this declaration, Python
        # treats refine_mode as a local variable inside on_key, leading to a bug where the
        # flag never changes and the refine branch is never used, causing new segmentations
        # to overwrite the existing mask rather than union with it.
        nonlocal current_seg, inst_id, skip_image, current_polygon, paint_mode, refine_mode
        key = event.key.lower()
        if key == 'y':
            # Accept current segmentation: assign to the result mask and clear.
            # Dualcurve refinement is handled in process_single_image, not here, to avoid nested event loops.
            if current_seg is not None and np.any(current_seg):
                mask_result[current_seg] = inst_id
                inst_id += 1
                current_seg = None
                current_polygon = []
                paint_mode = None
                update_overlay()
        elif key == 'r':
            # Reject current segmentation: clear segmentation and polygon
            current_seg = None
            current_polygon = []
            paint_mode = None
            update_overlay()
        elif key == 'q':
            # Finish annotation for this image
            plt.close(fig)
        elif key == 'x':
            # Skip this image entirely
            skip_image = True
            plt.close(fig)
        elif key == 'i':
            # Enter paint mode: the next polygon will be treated as an additive mask
            paint_mode = 'paint'
            current_polygon = []
            update_overlay()
        elif key == 'e':
            # Enter erase mode: the next polygon will be treated as a subtractive mask
            paint_mode = 'erase'
            current_polygon = []
            update_overlay()
        elif key == 'h':
            # Fill holes in the current segmentation using binary fill and morphological closing
            if current_seg is not None and np.any(current_seg):
                try:
                    # Fill holes using scipy.ndimage
                    filled = ndi.binary_fill_holes(current_seg).astype(bool)
                    # Apply a small closing to smooth boundaries and close small gaps
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    filled_u8 = filled.astype(np.uint8)
                    filled_u8 = cv2.morphologyEx(filled_u8, cv2.MORPH_CLOSE, kernel)
                    current_seg = (filled_u8 > 0)
                except Exception:
                    pass
                update_overlay()
        elif key == 'u':
            # Toggle refine mode.  In refine mode, the user draws a new polygon to
            # refine the current segmentation by unioning additional auto‑detected regions.
            refine_mode = not refine_mode
            # Exit any paint mode and clear current polygon when entering/exiting refine mode
            paint_mode = None
            current_polygon = []
            update_overlay()

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    # Disconnect callbacks
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    if skip_image:
        return None, True
    return mask_result, False

# Only execute the main function if this script is run as the top‑level
# program.  The call is placed here at the very end so that all helper
# functions are defined before main() is invoked.  This ensures that
# functions like auto_segment_region and annotate_image_auto_polygon are
# available when process_single_image calls them.
if __name__ == '__main__':
    main()