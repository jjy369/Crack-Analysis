from scipy.spatial.distance import cdist
from skimage.morphology import medial_axis, skeletonize_3d
from scipy.ndimage import convolve
from tqdm import tqdm
from sklearn.neighbors import KDTree
from dsepruning import skel_pruning_DSE
import os
import pandas as pd
from skimage.io import imread
from skimage import measure
import numpy as np
from skimage.measure import label, regionprops

# --------------------- ABI-related helper functions --------------------- #

def _wrap_to_90(angle_deg: float) -> float:
    """Wrap angle to [-90, 90] degrees (0° and 180° treated as equivalent)."""
    return (angle_deg + 90.0) % 180.0 - 90.0


def _delta_to_lane(theta_deg_array, lane_angle_deg: float):
    """
    Compute angle difference to lane direction Δθ in degrees, within [-90, 90].

    Parameters
    ----------
    theta_deg_array : float or array-like
        Crack orientation in degrees.
    lane_angle_deg : float
        Lane direction in degrees.
    """
    if np.isscalar(theta_deg_array):
        return _wrap_to_90(theta_deg_array - lane_angle_deg)
    # Vectorized version
    return (theta_deg_array + 90.0 - lane_angle_deg) % 180.0 - 90.0


def _classify_by_lane(delta_deg: float, tol_deg: float = 15.0) -> str:
    """
    Classify orientation relative to lane direction based on Δθ (deg).

    Returns
    -------
    'L' : longitudinal (|Δθ| <= tol_deg)
    'T' : transverse  (||Δθ| - 90| <= tol_deg)
    'O' : oblique     (otherwise)
    """
    d = float(delta_deg)
    if abs(d) <= tol_deg:
        return 'L'
    if abs(abs(d) - 90.0) <= tol_deg:
        return 'T'
    return 'O'


def _calculate_crack_length_for_ABI(skel: np.ndarray) -> float:
    """
    Skeleton length calculation used only for ABI / D_nonT.

    Uses 4 forward directions (right, down, down-right, up-right) to avoid
    double-counting edges.

    Parameters
    ----------
    skel : 2D ndarray
        Binary skeleton.

    Returns
    -------
    float
        Total skeleton length in pixel units.
    """
    skel = (skel > 0)
    H, W = skel.shape
    length = 0.0
    directions = [
        (0, 1, 1.0),             # right
        (1, 0, 1.0),             # down
        (1, 1, np.sqrt(2.0)),    # down-right
        (-1, 1, np.sqrt(2.0)),   # up-right
    ]
    for i in range(H):
        for j in range(W):
            if not skel[i, j]:
                continue
            for di, dj, d in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and skel[ni, nj]:
                    length += d
    return float(length)


def _region_length_by_skeleton(region, skel: np.ndarray) -> float:
    """
    Compute skeleton length within a connected region.

    Parameters
    ----------
    region : skimage.measure._regionprops.RegionProperties
        Connected region.
    skel : 2D ndarray
        Binary skeleton image.

    Returns
    -------
    float
        Total length of skeleton inside the region (pixel units).
    """
    mask = np.zeros_like(skel, dtype=bool)
    # region.coords is (row, col)
    mask[tuple(region.coords.T)] = True
    sub_skel = (skel > 0) & mask
    if np.count_nonzero(sub_skel) == 0:
        return 0.0
    return _calculate_crack_length_for_ABI(sub_skel)


# --------------------- ABI main functions --------------------- #

def compute_ABI(segmented_cracks: np.ndarray,
                pruned_skeleton: np.ndarray,
                lane_angle_deg: float = 90.0,
                pixel_size: float = 1.0,
                min_area: int = 5) -> float:
    """
    Compute Anisotropic Bias Index (ABI) in [-1, 1].

      +1 : longitudinal cracks dominate
      -1 : transverse cracks dominate
       0 : no strong directional bias (mixed orientations)

    Definition
    ----------
    Length-weighted mean( cos(2 * Δθ) )
      Δθ = θ_crack - θ_lane, wrapped into [-90, 90].

    Parameters
    ----------
    segmented_cracks : 2D ndarray
        Labeled crack image (integer labels) or binary mask (>0 is crack).
    pruned_skeleton : 2D ndarray
        Pruned skeleton (binary).
    lane_angle_deg : float, optional
        Lane direction in degrees.
    pixel_size : float, optional
        Physical size per pixel (e.g. m/px). If unknown, use 1.0 to get
        relative ABI.
    min_area : int, optional
        Minimum region area (in pixels) to be considered.

    Returns
    -------
    float
        ABI value in the range [-1, 1].
    """
    labeled = segmented_cracks if segmented_cracks.max() > 1 else \
              label(segmented_cracks, connectivity=2)

    thetas_deg = []
    lengths = []

    for region in regionprops(labeled):
        if region.area <= min_area:
            continue

        # skimage region.orientation in [-π/2, π/2], radians
        theta = np.degrees(region.orientation)
        theta = _wrap_to_90(theta)

        # Compute skeleton length inside this region
        L_px = _region_length_by_skeleton(region, pruned_skeleton)
        if L_px <= 0:
            continue

        thetas_deg.append(theta)
        lengths.append(L_px * pixel_size)

    if len(thetas_deg) == 0 or np.sum(lengths) <= 0:
        return 0.0

    delta = _delta_to_lane(np.array(thetas_deg, dtype=float), lane_angle_deg)
    ABI = float(
        np.sum(np.array(lengths) * np.cos(np.deg2rad(2.0 * delta))) /
        np.sum(lengths)
    )
    return ABI


def compute_D_nonT(segmented_cracks: np.ndarray,
                   pruned_skeleton: np.ndarray,
                   mask: np.ndarray,
                   lane_angle_deg: float = 90.0,
                   tol_deg: float = 15.0,
                   pixel_size: float = 1.0,
                   min_area: int = 5) -> float:
    """
    Non-transverse crack length density D_nonT.

    Definition
    ----------
      D_nonT = (total length of non-transverse cracks) / (image area)

    - "Non-transverse" = all classes except 'T' from _classify_by_lane.
    - Length is optionally scaled by pixel_size to become physical length.
    - Area = (#pixels) * pixel_size^2.

    Parameters
    ----------
    segmented_cracks : 2D ndarray
        Labeled crack image or binary mask.
    pruned_skeleton : 2D ndarray
        Pruned skeleton (binary).
    mask : 2D ndarray
        Crack mask (same size as skeleton).
    lane_angle_deg : float, optional
        Lane direction in degrees.
    tol_deg : float, optional
        Angular tolerance for L / T classification.
    pixel_size : float, optional
        Physical size per pixel (e.g. m/px).
    min_area : int, optional
        Minimum region area to be considered.

    Returns
    -------
    float
        Non-transverse crack length density.
    """
    labeled = segmented_cracks if segmented_cracks.max() > 1 else \
              label(segmented_cracks, connectivity=2)

    thetas_deg = []
    lengths = []

    for region in regionprops(labeled):
        if region.area <= min_area:
            continue

        theta = np.degrees(region.orientation)
        theta = _wrap_to_90(theta)
        L_px = _region_length_by_skeleton(region, pruned_skeleton)
        if L_px <= 0:
            continue

        thetas_deg.append(theta)
        lengths.append(L_px * pixel_size)

    if len(thetas_deg) == 0:
        return 0.0

    delta = _delta_to_lane(np.array(thetas_deg, dtype=float), lane_angle_deg)
    classes = np.array([_classify_by_lane(d, tol_deg) for d in delta])

    nonT_total_len = float(np.sum(np.array(lengths)[classes != 'T']))

    A = float(mask.shape[0] * mask.shape[1]) * (pixel_size ** 2)
    return (nonT_total_len / A) if A > 0 else 0.0


# --------------------- Crack geometry & topology functions --------------------- #

def detect_branch_points(skeleton, merge_threshold=5):
    """
    Detect branch points in a skeleton and merge nearby ones.

    Parameters
    ----------
    skeleton : 2D ndarray
        Binary skeleton image.
    merge_threshold : float, optional
        Distance threshold to merge nearby branch points.

    Returns
    -------
    ndarray of shape (N, 2)
        Coordinates of merged branch points (row, col).
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    branch_points = (neighbors > 12) & (skeleton > 0)

    branch_coords = np.column_stack(np.where(branch_points))

    if len(branch_coords) == 0:
        return np.empty((0, 2))

    distances = cdist(branch_coords, branch_coords)
    close_pairs = np.argwhere((distances < merge_threshold) & (distances > 0))

    merged_groups = {}
    assigned = {}
    cluster_id = 0

    for i, j in close_pairs:
        if i in assigned:
            group_id = assigned[i]
        elif j in assigned:
            group_id = assigned[j]
        else:
            group_id = cluster_id
            merged_groups[group_id] = []
            cluster_id += 1

        merged_groups[group_id].extend([i, j])
        assigned[i] = assigned[j] = group_id

    merged_branch_coords = []
    for group in merged_groups.values():
        merged_branch_coords.append(branch_coords[group][0])

    for i in range(len(branch_coords)):
        if i not in assigned:
            merged_branch_coords.append(branch_coords[i])

    return np.array(merged_branch_coords)


def extract_endpoints(skeleton, threshold=15):
    """
    Extract endpoints from a skeleton and suppress points that are too close.

    Parameters
    ----------
    skeleton : 2D ndarray
        Binary skeleton image.
    threshold : float, optional
        Minimum spacing between endpoints.

    Returns
    -------
    ndarray of shape (M, 2)
        Endpoint coordinates (row, col).
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = (neighbors == 11) & (skeleton > 0)

    endpoints = np.column_stack(np.where(endpoints))

    if len(endpoints) == 0:
        return endpoints

    distances = cdist(endpoints, endpoints)
    np.fill_diagonal(distances, np.inf)
    close_pairs = np.any(distances < threshold, axis=1)

    return endpoints[~close_pairs]


def segment_cracks(preprocessed_skeleton, nodes, min_size=10):
    """
    Remove node pixels from skeleton and label resulting crack segments.

    Parameters
    ----------
    preprocessed_skeleton : 2D ndarray
        Skeleton image.
    nodes : ndarray of shape (N, 2)
        Node coordinates (branch points and endpoints).
    min_size : int, optional
        Minimum segment area to keep.

    Returns
    -------
    labeled_cracks : 2D ndarray
        Labeled crack segments.
    int
        Maximum label (number of segments).
    """
    nodes = np.round(nodes).astype(int)
    notes_mask = np.zeros_like(preprocessed_skeleton, dtype=bool)

    for y, x in nodes:
        notes_mask[y, x] = True

    cleaned_skeleton = preprocessed_skeleton.copy()
    cleaned_skeleton[np.logical_and(preprocessed_skeleton, notes_mask)] = 0

    labeled_cracks = label(cleaned_skeleton, connectivity=2)

    for region in measure.regionprops(labeled_cracks):
        if region.area < min_size:
            labeled_cracks[labeled_cracks == region.label] = 0

    return labeled_cracks, np.max(labeled_cracks)


def compute_crack_roughness_from_mask(mask):
    """
    Compute crack roughness as perimeter / area.

    Parameters
    ----------
    mask : 2D ndarray
        Binary crack mask.

    Returns
    -------
    float
        Mean roughness over regions.
    """
    labeled_mask = label(mask > 0)
    regions = measure.regionprops(labeled_mask)

    roughness_list = []
    for region in regions:
        if region.area > 0:
            roughness_list.append(region.perimeter / region.area)

    return np.mean(roughness_list) if roughness_list else 0


def compute_bending_degree(segmented_cracks):
    """
    Compute bending degree (BD) for all crack segments.

    BD = sum_i(L_i) / sum_i(D_i),
    where L_i is the geodesic length along skeleton, D_i is straight-line
    distance between endpoints of segment i.
    """
    regions = measure.regionprops(segmented_cracks)

    L_sum = 0.0
    D_sum = 0.0

    for region in regions:
        coords = region.coords
        if len(coords) < 2:
            continue

        L_i = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
        (x_s, y_s), (x_e, y_e) = coords[0], coords[-1]

        D_i = np.sqrt((x_s - x_e)**2 + (y_s - y_e)**2)

        L_sum += L_i
        D_sum += D_i

    return L_sum / D_sum if D_sum > 0 else 1.0


# ---------- Crack width helpers ---------- #

def SVD(points):
    """SVD helper for normal estimation."""
    pts = points.copy()
    c = np.mean(pts, axis=0)
    A = pts - c
    A = A.T
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
    normal = u[:, -1]

    nlen = np.sqrt(np.dot(normal, normal))
    normal = normal / nlen
    return u, s, c, normal


def estimate_normals(points, n):
    """Estimate normals using kNN and SVD (original version)."""
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(pts, k=n, return_distance=False, dualtree=False, breadth_first=False)
    normals = []
    for i in range(pts.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def get_crack_ctrlpts(centers, normals, bpoints, hband=5, vband=2, est_width=0, image_shape=None):
    """
    Original crack control-points & width extraction.

    Parameters
    ----------
    centers : ndarray
        Skeleton center points (N x 2).
    normals : ndarray
        Normals at centers (N x 2).
    bpoints : ndarray
        Boundary points of cracks.
    hband : float
        Horizontal search window in local coordinates.
    vband : float
        Unused but kept for compatibility.
    est_width : float
        Estimated width to refine search band.
    image_shape : tuple
        Image shape, unused but kept for compatibility.

    Returns
    -------
    interp_segm : ndarray
        Interpolated segment endpoints in original coordinates.
    widths : ndarray
        Width information per center (index, top_offset, bottom_offset).
    """
    cpoints = np.copy(centers)
    cnormals = np.copy(normals)

    xmatrix = np.array([[0, 1], [-1, 0]])
    cnormalsx = np.dot(xmatrix, cnormals.T).T
    N = cpoints.shape[0]

    interp_segm = []
    widths = []
    for i in range(N):
        try:
            ny = cnormals[i]
            nx = cnormalsx[i]
            tform = np.array([nx, ny])

            # transform boundary points & center points
            bpoints_loc = np.dot(tform, bpoints.T).T
            cpoints_loc = np.dot(tform, cpoints.T).T
            ci = cpoints_loc[i]

            # horizontal window around the center in local coords
            bl_ind = (bpoints_loc[:, 0] - (ci[0] - hband)) * (bpoints_loc[:, 0] - ci[0]) < 0
            br_ind = (bpoints_loc[:, 0] - ci[0]) * (bpoints_loc[:, 0] - (ci[0] + hband)) <= 0
            bl = bpoints_loc[bl_ind]
            br = bpoints_loc[br_ind]

            if est_width > 0:
                half_est_width = est_width / 2
                blt = bl[(bl[:, 1] - (ci[1] + half_est_width)) * (bl[:, 1] - ci[1]) < 0]
                blb = bl[(bl[:, 1] - (ci[1] - half_est_width)) * (bl[:, 1] - ci[1]) < 0]
                brt = br[(br[:, 1] - (ci[1] + half_est_width)) * (br[:, 1] - ci[1]) < 0]
                brb = br[(br[:, 1] - (ci[1] - half_est_width)) * (br[:, 1] - ci[1]) < 0]
            else:
                blt = bl[bl[:, 1] > np.mean(bl[:, 1])]
                blb = bl[bl[:, 1] < np.mean(bl[:, 1])]
                brt = br[br[:, 1] > np.mean(br[:, 1])]
                brb = br[br[:, 1] < np.mean(br[:, 1])]

            # top & bottom points on left/right side
            t1 = blt[np.argsort(blt[:, 0])[-1]]
            t2 = brt[np.argsort(brt[:, 0])[0]]
            b1 = blb[np.argsort(blb[:, 0])[-1]]
            b2 = brb[np.argsort(brb[:, 0])[0]]

            # linear interpolation to get crack boundary at x = ci[0]
            interp1 = (ci[0] - t1[0]) * ((t2[1] - t1[1]) / (t2[0] - t1[0])) + t1[1]
            interp2 = (ci[0] - b1[0]) * ((b2[1] - b1[1]) / (b2[0] - b1[0])) + b1[1]

            # ensure one is above the center and the other below (forming a width)
            if interp1 - ci[1] > 0 and interp2 - ci[1] < 0:
                widths.append([i, interp1 - ci[1], interp2 - ci[1]])

                interps = np.array([[ci[0], interp1], [ci[0], interp2]])
                interps_rec = np.dot(np.linalg.inv(tform), interps.T).T
                interps_rec = interps_rec.reshape(1, -1)[0, :]
                interp_segm.append(interps_rec)
        except Exception:
            continue

    interp_segm = np.array(interp_segm)
    widths = np.array(widths)
    return interp_segm, widths


def compute_crack_width(mask, skeleton):
    """
    Compute mean crack width (in pixels) using the original geometric method.

    Parameters
    ----------
    mask : 2D ndarray
        Binary crack mask.
    skeleton : 2D ndarray
        Binary skeleton.

    Returns
    -------
    float
        Mean crack width in pixels.
    """
    x, y = np.where(skeleton > 0)
    centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    normals = estimate_normals(centers, 9)

    # boundary contours from mask
    contours = measure.find_contours(mask, 0.5)

    interp_segm, widths = get_crack_ctrlpts(
        centers,
        normals,
        np.vstack(contours),
        hband=2,
        vband=2,
        est_width=15,
        image_shape=mask.shape
    )

    if widths.size > 0:
        crack_widths = np.abs(widths[:, 1] - widths[:, 2])
        avg_width = np.mean(crack_widths)
        return avg_width
    else:
        return 0.0


def calculate_crack_length(skel):
    """
    Compute total skeleton length using 8-connectivity.

    Parameters
    ----------
    skel : 2D ndarray
        Binary skeleton image.

    Returns
    -------
    float
        Total length in pixel units.
    """
    length = 0.0
    h, w = skel.shape
    for i in range(h):
        for j in range(w):
            if skel[i, j]:
                for di, dj, d in [
                    (-1, 0, 1), (1, 0, 1),
                    (0, -1, 1), (0, 1, 1),
                    (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
                    (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))
                ]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and skel[ni, nj]:
                        length += d
    return length / 2.0


# ====================== MAIN LOOP ====================== #

input_folder = r"F:\GPR\DSE-skeleton-pruning\temp"
output_excel_path = r"F:\GPR\数据\crack_analysis_results_test2.xlsx"

results = []

for image_name in tqdm(os.listdir(input_folder)):
    if not (image_name.endswith(".png") or image_name.endswith(".jpg")):
        continue

    mask = imread(os.path.join(input_folder, image_name))
    H, W = mask.shape[:2]

    # Each image corresponds to 1 m²: pixel size in meters
    pixel_size_m = (1.0 / (H * W)) ** 0.5

    # Crack roughness from original mask
    roughness = compute_crack_roughness_from_mask(mask)

    # Skeleton for BD / width / complexity
    skel = skeletonize_3d(mask)
    dist_ma = medial_axis(mask)
    pruned_skel = skel_pruning_DSE(skel, dist_ma, 50)

    branch_points = detect_branch_points(pruned_skel)
    endpoints = extract_endpoints(pruned_skel)

    nodes = np.vstack((branch_points, endpoints)) if (len(branch_points) + len(endpoints)) > 0 else np.empty((0, 2))
    segmented_cracks, _ = segment_cracks(pruned_skel, nodes)

    # Bending degree
    BD = compute_bending_degree(segmented_cracks)

    # Crack length (m) and width (mm)
    width_pixel = compute_crack_width(mask, pruned_skel)
    length_pixel = calculate_crack_length(pruned_skel)

    length_m = length_pixel * pixel_size_m
    width_mm = width_pixel * pixel_size_m * 1000.0

    # Complexity index based on branch/end counts
    B_count = len(branch_points)
    N_end = len(endpoints)
    N_nodes = B_count + N_end
    complexity_index = (B_count - N_end) / N_nodes if N_nodes > 0 else 0.0

    # ABI (using original implementation rules)
    mask_bin = (mask > 0).astype(np.uint8)
    skel_abi = skeletonize_3d(mask_bin)
    ABI_value = compute_ABI(
        segmented_cracks=label(mask_bin),
        pruned_skeleton=(skel_abi > 0).astype(np.uint8),
        lane_angle_deg=0.0,
        pixel_size=1.0,
        min_area=5,
    )

    # ---- Normalization constants (fixed) ----
    L_Q05, L_Q95 = 1.112, 3.817
    W_Q05, W_Q95 = 2.369, 6.870
    BD_Q05, BD_Q95 = 0.754, 5.975
    R_Q05, R_Q95 = 0.040, 0.174

    I_norm = (length_m - L_Q05) / (L_Q95 - L_Q05)
    W_norm = (width_mm - W_Q05) / (W_Q95 - W_Q05)
    BD_norm = (np.log(BD + 1.0) - BD_Q05) / (BD_Q95 - BD_Q05)
    R_norm = (roughness - R_Q05) / (R_Q95 - R_Q05)
    ABI_norm = (ABI_value + 1.0) / 2.0
    Complexity_norm = (complexity_index + 1.0) / 2.0

    # Crack Condition Index (CCI)
    CCI = (
        0.183 * I_norm +
        0.147 * W_norm +
        0.164 * BD_norm +
        0.139 * R_norm +
        0.190 * ABI_norm +
        0.176 * Complexity_norm
    )

    results.append({
        "Image_Name": image_name,
        "I_norm": I_norm,
        "W_norm": W_norm,
        "BD_norm": BD_norm,
        "R_norm": R_norm,
        "ABI_norm": ABI_norm,
        "Complexity_norm": Complexity_norm,
        "CCI": CCI,
    })

# ---------------- Save Results ---------------- #

results_df = pd.DataFrame(results).round(3)
results_df.to_excel(output_excel_path, index=False)

print(f"Results saved to Excel: {output_excel_path}")
