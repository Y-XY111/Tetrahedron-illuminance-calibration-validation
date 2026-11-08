'Standalone HDR photometric metrics and Excel export.\n\nSelect a panorama HDR, choose one calibration method and select an Excel\noutput path. Enter measured panoramic illuminance (lx) to derive k, or\nenter a known k directly. These inputs are mutually exclusive.\nIlluminance and luminance include k; dimensionless CV and diffuseness do not\nreceive an additional multiplier. Uses CPU algorithms without the main app,\nQt, plotting libraries or CuPy. Sampling and orientation are preserved.\n'
from pathlib import Path
import argparse
import math
import time
from functools import lru_cache
from scipy.special import factorial
import cv2
import numpy as np

k = 1.0
_HDR_CACHE = {}


def _read_hdr(filepath, flags=None):
    path=Path(filepath).expanduser().resolve()
    key=str(path)
    if key not in _HDR_CACHE:
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError('Cannot read HDR: %s' % path)
        if image.ndim!=3 or image.shape[2]!=3 or not np.issubdtype(image.dtype,np.floating):
            raise ValueError('A three-channel floating-point HDR is required; JPG/PNG previews cannot be used')
        h,w=image.shape[:2]
        if abs(w/h-2)>0.02 or h<80:
            raise ValueError('A 2:1 equirectangular HDR at least 80 pixels high is required; stitch fisheyes first')
        if not np.all(np.isfinite(image)) or np.any(image<0):
            raise ValueError('The HDR contains invalid or negative pixel values')
        _HDR_CACHE.clear()
        _HDR_CACHE[key]=image
    return _HDR_CACHE[key]


def _is_cupy_backend(xp):
    return False


def _backend_array_to_numpy(array,xp):
    return np.asarray(array)


def _pairwise_block_rows(n_cols,xp,dtype=np.float64):
    if n_cols<=0:return 1
    budget=256*1024*1024
    rows=budget//max(1,n_cols*(np.dtype(dtype).itemsize*4+4))
    return max(64,min(int(rows),4096,n_cols))


def _target_illuminance_auto(target_vecs,x_y_z,P,pix_source_R,context):
    return _target_illuminance_backend(target_vecs,x_y_z,P,pix_source_R,context,np)

def _resolve_calibration_factor(calibration_factor):
    """Snapshot legacy k only when the caller has not supplied its own factor."""
    factor = float(k if calibration_factor is None else calibration_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError('The luminance calibration factor must be finite and greater than zero')
    return factor


def _equirectangular_solid_angle_map(height, width, dtype=np.float64):
    """
    Return the exact per-pixel solid angle for a 2:1 equirectangular panorama.

    For row y:
        ΔΩ_y = (sin λ_upper - sin λ_lower) · Δφ

    Every pixel in the same row has the same solid angle.  The complete map
    sums to 4π sr (up to floating-point roundoff).
    """
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError('Image dimensions must be positive integers')
    if abs(width / float(height) - 2.0) > 0.02:
        raise ValueError(f'A 2:1 equirectangular panorama is required; current size: {width}×{height}')
    row_edges = np.arange(height + 1, dtype=np.float64)
    latitude_edges = np.pi / 2.0 - row_edges * np.pi / height
    latitude_upper = latitude_edges[:-1]
    latitude_lower = latitude_edges[1:]
    delta_phi = 2.0 * np.pi / width
    row_solid_angle = (np.sin(latitude_upper) - np.sin(latitude_lower)) * delta_phi
    domega = np.broadcast_to(row_solid_angle[:, None], (height, width)).copy()
    return domega.astype(dtype, copy=False)


def _pairwise_dot3(left, right):
    return left[:, 0:1] * right[None, :, 0] + left[:, 1:2] * right[None, :, 1] + left[:, 2:3] * right[None, :, 2]


def _target_illuminance_backend(target_vecs, x_y_z, P, pix_source_R, context, xp, dtype=np.float64):
    targets_np = np.asarray(target_vecs, dtype=dtype).reshape(-1, 3)
    n_targets = targets_np.shape[0]
    n = int(np.prod(P.shape))
    targets = xp.asarray(targets_np, dtype=dtype)
    V = xp.asarray(x_y_z.reshape(n, 3), dtype=dtype)
    L = xp.asarray(P.reshape(n), dtype=dtype)
    norms = xp.linalg.norm(V, axis=1) + 1e-12
    target_norms = xp.linalg.norm(targets, axis=1) + 1e-12
    out = xp.empty(n_targets, dtype=dtype)
    block_rows = _pairwise_block_rows(n, xp, dtype=dtype)
    for start_idx in range(0, n_targets, block_rows):
        end_idx = min(start_idx + block_rows, n_targets)
        Tb = targets[start_idx:end_idx]
        dot = _pairwise_dot3(Tb, V)
        cos_mat = dot / (target_norms[start_idx:end_idx, None] * norms[None, :])
        out[start_idx:end_idx] = (xp.clip(cos_mat, 0.0, None) * L[None, :]).sum(axis=1) / pix_source_R ** 2
    if _is_cupy_backend(xp):
        xp.cuda.Stream.null.synchronize()
    return np.asarray(_backend_array_to_numpy(out, xp), dtype=np.float64)


def Field_luminance_CV(filepath, *, calibration_factor=None):
    """
    Return the whole-sphere luminance CV using exact per-pixel solid angles.

    This scalar function now uses the same weighting convention as
    luminance_uniformity(); the previous cone-shaped resampling is removed.
    """
    factor = _resolve_calibration_factor(calibration_factor)
    start = time.time()
    img1 = _read_hdr(filepath, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img1 is None:
        raise FileNotFoundError(f'Cannot read HDR file: {filepath}')
    if abs(img1.shape[1] / img1.shape[0] - 2.0) > 0.02:
        raise ValueError(f'A 2:1 equirectangular HDR is required; current size: {img1.shape[1]}×{img1.shape[0]}')
    target_h = max(1, round(img1.shape[0] * 0.025))
    img = cv2.resize(img1, (2 * target_h, target_h), interpolation=cv2.INTER_NEAREST)
    (r, c) = img.shape[:2]
    if abs(c / r - 2.0) > 0.02:
        raise RuntimeError(f'The working image is not 2:1: {c}×{r}')
    b = img[..., 0].astype(np.float64)
    g = img[..., 1].astype(np.float64)
    rr = img[..., 2].astype(np.float64)
    luminance = (0.0722 * b + 0.7152 * g + 0.2126 * rr) * 179.0 * factor
    domega = _equirectangular_solid_angle_map(r, c, dtype=np.float64)
    valid = np.isfinite(luminance) & (luminance >= 0) & np.isfinite(domega)
    if not np.any(valid):
        return 0.0
    values = luminance[valid]
    weights = domega[valid]
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if weight_sum <= 0:
        return 0.0
    mean_l = float(np.sum(values * weights, dtype=np.float64) / weight_sum)
    second_moment = float(np.sum(values * values * weights, dtype=np.float64) / weight_sum)
    variance = max(second_moment - mean_l * mean_l, 0.0)
    std_l = math.sqrt(variance)
    cv_l = std_l / mean_l if mean_l > 0 else 0.0
    print(f'Processed image size: {r}×{c}')
    print(f'Total exact solid angle: {weight_sum:.12g} sr')
    print(f'Luminance CV calculation time: {time.time() - start:.2f} s')
    return float(cv_l)


def spherical_harmonic_decomposition(filepath, entered_panoramic_E=None, *, calibration_factor=None):
    """Calculate the first-order spherical-harmonic lighting metrics.

    ``entered_panoramic_E`` keeps the legacy calibration workflow: when the
    module-wide factor has not yet been calibrated, the measured panoramic
    illuminance is used to derive it.  Glare evaluation already owns an
    independently calculated calibration factor, so it can pass that factor
    explicitly without recalibrating from the shielded HDR.
    """
    global k
    if calibration_factor is not None:
        applied_calibration = float(calibration_factor)
        if not np.isfinite(applied_calibration) or applied_calibration <= 0.0:
            raise ValueError('The luminance calibration factor must be finite and greater than zero')
    else:
        if entered_panoramic_E is None:
            raise ValueError('Provide measured panoramic illuminance or a luminance calibration factor')
        entered_panoramic_E = float(entered_panoramic_E)
        if not np.isfinite(entered_panoramic_E) or entered_panoramic_E <= 0.0:
            raise ValueError('Panoramic illuminance must be a finite number greater than zero')
        applied_calibration = None
    start = time.time()
    img = _read_hdr(filepath, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    img = cv2.resize(img, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
    r = img.shape[0]
    c = img.shape[1]
    pix_source_R = r / math.pi
    luminance_map = np.zeros((r, c))
    b_channel = img[..., 0]
    g_channel = img[..., 1]
    r_channel = img[..., 2]
    luminance_map = (0.0722 * b_channel + 0.7152 * g_channel + 0.2126 * r_channel) * 179
    pix_source_R = r / math.pi
    luminance_matrix = np.zeros((r, c, 3))
    P = np.zeros((r, c))
    theta_s = np.linspace(0, math.pi, r, endpoint=False)
    phi_s = 2 * math.pi / c * (c - np.arange(c))
    theta_s_2d = theta_s.reshape(-1, 1)
    phi_s_2d = phi_s.reshape(1, -1)
    sin_theta = np.sin(theta_s_2d)
    luminance = luminance_map * sin_theta
    luminance_matrix = np.stack([np.broadcast_to(theta_s_2d, (r, c)), np.broadcast_to(phi_s_2d, (r, c)), luminance], axis=2)
    P = luminance.copy()
    ll = [0, 1, 1, 1, 2, 2, 2, 2, 2]
    mm = [0, -1, 0, 1, -2, -1, 0, 1, 2]
    ylm = np.zeros((r, c, 9))
    f_value = np.zeros((r, c, 4))

    @lru_cache(maxsize=None)
    def factorial_cache(n):
        return factorial(n)

    @lru_cache(maxsize=None)
    def double_factorial(n):
        'Calculate the double factorial n!!.'
        if n <= 0:
            return 1
        result = 1
        for i in range(n, 0, -2):
            result *= i
        return result

    def vectorized_p_l_m(l, m, x):
        'Calculate vectorized associated Legendre polynomials.'
        x = np.asarray(x)
        m_abs = abs(m)
        n = l
        p_m_m = (-1) ** m_abs * double_factorial(2 * m_abs - 1) * (1 - x ** 2) ** (m_abs / 2)
        if n == m_abs:
            return p_m_m
        p_m1_m = x * (2 * m_abs + 1) * p_m_m
        if n == m_abs + 1:
            return p_m1_m
        p_prev_prev = p_m_m
        p_prev = p_m1_m
        for i in range(m_abs + 2, n + 1):
            p_current = (x * (2 * i - 1) * p_prev - (i + m_abs - 1) * p_prev_prev) / (i - m_abs)
            (p_prev_prev, p_prev) = (p_prev, p_current)
        return p_prev

    def vectorized_funck(l, m):
        'Calculate vectorized normalization coefficients.'
        m_abs = abs(m)
        return np.sqrt((2 * l + 1) * factorial_cache(l - m_abs) / (factorial_cache(l + m_abs) * 4 * np.pi))

    def vectorized_y_l_m(l, m, theta, phi):
        'Calculate vectorized spherical harmonics.'
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        costheta = np.cos(theta)
        if m > 0:
            return np.sqrt(2) * vectorized_funck(l, m) * np.cos(m * phi) * vectorized_p_l_m(l, m, costheta)
        elif m < 0:
            m_abs = -m
            return np.sqrt(2) * vectorized_funck(l, m_abs) * np.sin(m_abs * phi) * vectorized_p_l_m(l, m_abs, costheta)
        else:
            return vectorized_funck(l, 0) * vectorized_p_l_m(l, 0, costheta)

    def compute_spherical_harmonics(luminance_matrix, P, max_degree=3):
        'Calculate spherical harmonics and weighted values in batches.\n\nReturns ylm with shape (r, c, 9) and f_value with shape (r, c, 4).\n'
        ylm = np.zeros((r, c, 9), dtype=np.float64)
        f_value = np.zeros((r, c, 4), dtype=np.float64)
        theta = luminance_matrix[..., 0]
        phi = luminance_matrix[..., 1]
        for l_idx in range(len(ll)):
            l = ll[l_idx]
            m = mm[l_idx]
            if l > max_degree:
                continue
            ylm[..., l_idx] = vectorized_y_l_m(l, m, theta, phi)
            if l_idx < 4:
                f_value[..., l_idx] = ylm[..., l_idx] * P
        return (ylm, f_value)
    (ylm, f_value) = compute_spherical_harmonics(luminance_matrix, P, max_degree=3)
    fn = np.sum(f_value * (1 / (pix_source_R * pix_source_R)), axis=(0, 1))
    Scalar = math.sqrt(math.pi) / 2 * fn[0]
    if calibration_factor is not None:
        applied_calibration = float(calibration_factor)
    elif k == 1:
        k = entered_panoramic_E / Scalar
        applied_calibration = k
    else:
        applied_calibration = k
    Scalar_calibrated = Scalar * applied_calibration
    fistorder_strength = math.sqrt(fn[1] * fn[1] + fn[2] * fn[2] + fn[3] * fn[3])
    first_zero = math.sqrt(fn[1] * fn[1] + fn[2] * fn[2] + fn[3] * fn[3]) / fn[0]
    Normalized_first_zero = 1 - first_zero / math.sqrt(3)
    diffuseness = Normalized_first_zero
    y = -fn[1] * math.sqrt(4 * math.pi / 3) * applied_calibration
    z = fn[2] * math.sqrt(4 * math.pi / 3) * applied_calibration
    x = -fn[3] * math.sqrt(4 * math.pi / 3) * applied_calibration
    vector_norm = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    z1 = float(np.clip(z / vector_norm, -1, 1)) if vector_norm > 0 else 1.0
    theta_vector = math.acos(z1) / math.pi * 180
    phi_vector = (math.atan2(y, x) / math.pi + 1) * 180
    end = time.time()
    print(f'Spherical harmonic calculation time: {end - start:.2f} s')
    return (theta_vector, phi_vector, diffuseness, Scalar_calibrated, np.array([x, y, z]))


def illuminance_variation_coefficient(filepath, *, calibration_factor=None):
    factor = _resolve_calibration_factor(calibration_factor)
    start = time.time()
    img1 = _read_hdr(filepath, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    img = cv2.resize(img1, (0, 0), fx=0.025, fy=0.025, interpolation=cv2.INTER_NEAREST)
    r = img.shape[0]
    c = img.shape[1]
    pix_source_R = r / math.pi
    luminance_map = np.zeros((r, c))
    b_channel = img[..., 0]
    g_channel = img[..., 1]
    r_channel = img[..., 2]
    luminance_map = (0.0722 * b_channel + 0.7152 * g_channel + 0.2126 * r_channel) * 179 * factor
    pix_source_R = r / math.pi
    luminance_matrix = np.zeros((r, c, 3))
    x_y_z = np.zeros((r, c, 3))
    P = np.zeros((r, c))
    theta_s = np.linspace(0, math.pi, r, endpoint=False)
    phi_s = np.linspace(2 * math.pi, 0, c, endpoint=False)
    (theta_grid, phi_grid) = np.meshgrid(theta_s, phi_s, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    sin_theta = np.sin(theta_grid)
    luminance = np.float64(luminance_map * sin_theta)
    luminance_matrix = np.stack([theta_grid, phi_grid, luminance], axis=-1)
    x_y_z = np.stack([x, y, z], axis=-1)
    P = luminance.copy()
    E = np.zeros((r, c), dtype=np.float64)
    out_img = np.zeros((r, c), dtype=np.float64)
    valid_mask = np.zeros((r, c), dtype=bool)
    center = (c - 1) / 2.0
    theta_rows = np.arange(r, dtype=np.float64) / r * math.pi
    effective_c = np.sin(theta_rows) * pix_source_R * 2.0 * math.pi
    c_range = np.maximum(1, np.rint(effective_c)).astype(int)
    center = (c - 1) / 2.0
    valid_mask = np.zeros((r, c), dtype=bool)
    j_starts = np.ceil(center - c_range / 2.0).astype(int)
    j_ends = j_starts + c_range
    j_starts = np.clip(j_starts, 0, c)
    j_ends = np.clip(j_ends, 0, c)
    for i in range(r):
        if j_ends[i] > j_starts[i]:
            valid_mask[i, j_starts[i]:j_ends[i]] = True
    tgt_flat_indices = []
    theta_list = []
    phi_list = []
    for i in range(r):
        (js0, js1) = (j_starts[i], j_ends[i])
        if js1 <= js0:
            continue
        rng = np.arange(js0, js1, dtype=np.float64)
        phi_s_row = (js1 - rng) / c_range[i] * (2.0 * math.pi)
        m = rng.size
        theta_list.append(np.full(m, theta_rows[i], dtype=np.float64))
        phi_list.append(phi_s_row)
        tgt_flat_indices.append((i * c + rng).astype(np.int64))
    if len(theta_list) == 0:
        out_img = np.zeros((r, c), dtype=np.float64)
        E = out_img[valid_mask]
    else:
        theta_t = np.concatenate(theta_list)
        phi_t = np.concatenate(phi_list)
        tgt_idx = np.concatenate(tgt_flat_indices)
        M = theta_t.shape[0]
        st = np.sin(theta_t)
        V_t = np.stack([st * np.cos(phi_t), st * np.sin(phi_t), np.cos(theta_t)], axis=1)
        acc = _target_illuminance_auto(V_t, x_y_z, P, pix_source_R, context='illuminance_variation_coefficient')
        out_img = np.zeros((r, c), dtype=np.float64)
        out_img.reshape(-1)[tgt_idx] = acc
        E = out_img[valid_mask].astype(np.float64)
    e_avg = np.mean(E)
    e_sd = np.std(E)
    cvE = e_sd / e_avg
    end = time.time()
    print(f'Illuminance CV calculation time: {end - start:.2f} s')
    return cvE


def various_illuminance_calculation(xyz, filepath, *, calibration_factor=None):
    factor = _resolve_calibration_factor(calibration_factor)
    start = time.time()
    img1 = _read_hdr(filepath, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    img = cv2.resize(img1, (0, 0), fx=0.025, fy=0.025, interpolation=cv2.INTER_NEAREST)
    (x, y, z) = xyz
    r = img.shape[0]
    c = img.shape[1]
    a = math.atan2(y, x)
    if a > -math.pi / 2 and a <= math.pi:
        a = a - math.pi / 2
    elif a <= -math.pi / 2 and a > -math.pi:
        a = a + math.pi * 3 / 2
    u = (1 - (1 + a / math.pi) / 2) * c
    z1 = z / np.linalg.norm(xyz)
    v = math.acos(z1) / math.pi * r
    u = max(0, min(u, c - 1))
    v = max(0, min(v, r - 1))
    pix_source_R = r / math.pi
    luminance_map = np.zeros((r, c))
    b_channel = img[..., 0]
    g_channel = img[..., 1]
    r_channel = img[..., 2]
    luminance_map = (0.0722 * b_channel + 0.7152 * g_channel + 0.2126 * r_channel) * 179 * factor
    luminance_matrix = np.zeros((r, c, 3))
    x_y_z = np.zeros((r, c, 3))
    P = np.zeros((r, c))
    theta_s = np.linspace(0, math.pi, r, endpoint=False)
    phi_s = np.linspace(2 * math.pi, 0, c, endpoint=False)
    (theta_grid, phi_grid) = np.meshgrid(theta_s, phi_s, indexing='ij')
    sin_theta = np.sin(theta_grid)
    luminance = np.float64(luminance_map * sin_theta)
    luminance_matrix = np.stack([theta_grid, phi_grid, luminance], axis=-1)
    sin_theta = np.sin(theta_grid)
    cos_theta = np.cos(theta_grid)
    cos_phi = np.cos(phi_grid)
    sin_phi = np.sin(phi_grid)
    x_y_z = np.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=-1)
    P = luminance.copy()
    v_idx = int(v) - 1
    u_idx = int(u) - 1
    v_idx = max(0, min(v_idx, r - 1))
    u_idx = max(0, min(u_idx, c - 1))
    angles_deg = (0, 45, 90, 135, 180, 225, 270, 315)
    phi = np.deg2rad(np.array(angles_deg, dtype=np.float64))
    vertical_targets = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
    vertical_targets /= np.linalg.norm(vertical_targets, axis=1, keepdims=True) + 1e-12
    target_vecs = np.vstack([x_y_z[0, 0].reshape(1, 3), x_y_z[v_idx, u_idx].reshape(1, 3), x_y_z[int(r / 2)].reshape(c, 3), vertical_targets])
    target_E = _target_illuminance_auto(target_vecs, x_y_z, P, pix_source_R, context='various_illuminance_calculation')
    E_hor = float(target_E[0])
    E_vector = float(target_E[1])
    E1 = target_E[2:2 + c]
    E_cylinder = float(np.mean(E1)) if E1.size else 0.0
    E_vals = target_E[2 + c:2 + c + len(angles_deg)]
    end = time.time()
    print(f'Directional illuminance calculation time: {end - start:.2f} s')
    return (E_hor, E_vals, E_cylinder, E_vector)


def compute_40deg_horizontal_band_luminance(hdr_path, *, calibration_factor=None):
    'Return solid-angle-weighted mean luminance in the horizontal 40-degree band.\n\nInput: a 2:1 equirectangular HDR panorama. The band covers zenith angles\n70 to 110 degrees. The returned luminance includes k and is in cd/m^2.\n'
    factor = _resolve_calibration_factor(calibration_factor)
    img = _read_hdr(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read HDR image: {hdr_path}')
    if img.ndim == 3 and img.shape[2] == 3:
        (B, G, R) = cv2.split(img.astype(np.float64))
        L = (0.2126 * R + 0.7152 * G + 0.0722 * B) * 179 * factor
    else:
        L = img.astype(np.float64)
    (H, W) = L.shape
    y = np.arange(H)
    theta = np.pi * (y + 0.5) / H
    (theta1_deg, theta2_deg) = (70.0, 110.0)
    theta1 = np.deg2rad(theta1_deg)
    theta2 = np.deg2rad(theta2_deg)
    band_mask_rows = (theta >= theta1) & (theta <= theta2)
    if not np.any(band_mask_rows):
        raise RuntimeError('No rows fall into the 40° band. Check projection / image size.')
    weights_rows = np.sin(theta[band_mask_rows])
    L_row_sum = L[band_mask_rows, :].sum(axis=1)
    num = np.sum(L_row_sum * weights_rows)
    den = np.sum(weights_rows) * W
    L_band = num / (den + 1e-12)
    return L_band


def calculate_panorama_calibration_factor(filepath, entered_panoramic_E, resize_scale=0.05):
    'Derive k from the current HDR and measured panoramic illuminance.\n\nDoes not read or modify the module-wide k. Returns the calibration factor\nand the uncalibrated HDR panoramic illuminance, in that order.\n'
    measured_illuminance = float(entered_panoramic_E)
    if not np.isfinite(measured_illuminance) or measured_illuminance <= 0:
        raise ValueError('Measured panoramic illuminance must be a finite positive number')
    scale = float(resize_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('The image resize scale must be a finite positive number')
    img = _read_hdr(filepath, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {filepath}')
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    (rows, cols) = img.shape[:2]
    if rows <= 0 or cols <= 0:
        raise ValueError('Invalid HDR image dimensions')
    b_channel = img[..., 0].astype(np.float64)
    g_channel = img[..., 1].astype(np.float64)
    r_channel = img[..., 2].astype(np.float64)
    luminance_map = (0.0722 * b_channel + 0.7152 * g_channel + 0.2126 * r_channel) * 179.0
    theta = np.linspace(0.0, math.pi, rows, endpoint=False)
    weighted_luminance = luminance_map * np.sin(theta)[:, np.newaxis]
    y_0_0 = 1.0 / math.sqrt(4.0 * math.pi)
    pixel_radius = rows / math.pi
    coefficient_0 = np.sum(y_0_0 * weighted_luminance) / (pixel_radius * pixel_radius)
    raw_panoramic_illuminance = math.sqrt(math.pi) / 2.0 * coefficient_0
    if not np.isfinite(raw_panoramic_illuminance) or raw_panoramic_illuminance <= 0:
        raise ValueError('Cannot calculate valid panoramic illuminance from this HDR')
    calibration_factor = measured_illuminance / raw_panoramic_illuminance
    if not np.isfinite(calibration_factor) or calibration_factor <= 0:
        raise ValueError('Cannot derive a valid calibration factor')
    return (float(calibration_factor), float(raw_panoramic_illuminance))


def _compute_metrics_at_factor(input_path, calibration_factor, progress=None):
    """Compute final platform metrics at the explicitly supplied factor."""
    progress = progress or (lambda message: print(message, flush=True))
    path = Path(input_path).expanduser().resolve()
    image = _read_hdr(path)
    if not np.any(image > 0):
        raise ValueError('An entirely black HDR has zero mean luminance; its coefficient of variation is undefined')
    shape = image.shape
    progress('1/5 Calculating panoramic illuminance and diffuseness...')
    (_, _, diffuseness, scalar, xyz) = spherical_harmonic_decomposition(str(path), calibration_factor=calibration_factor)
    progress('2/5 Calculating illuminance coefficient of variation...')
    cv_e = illuminance_variation_coefficient(str(path), calibration_factor=calibration_factor)
    progress('3/5 Calculating horizontal, vertical and cylindrical illuminance...')
    safe_xyz = xyz if np.linalg.norm(xyz) > max(abs(scalar), 1.0) * 1e-12 else np.array([0.0, 0.0, 1.0])
    (horizontal, vertical, cylinder, _) = various_illuminance_calculation(safe_xyz, str(path), calibration_factor=calibration_factor)
    progress('4/5 Calculating luminance coefficient of variation...')
    cv_l = Field_luminance_CV(str(path), calibration_factor=calibration_factor)
    progress('5/5 Calculating 40-degree horizontal band luminance...')
    band_l = compute_40deg_horizontal_band_luminance(str(path), calibration_factor=calibration_factor)
    rows = [('Horizontal illuminance', horizontal, 'lx', True, 'Illuminance on a horizontal plane facing the zenith'), ('Vertical illuminance (180 degrees)', vertical[4], 'lx', True, 'Illuminance at 180 degrees; other directions are listed separately'), ('Cylindrical illuminance', cylinder, 'lx', True, 'Circumferential mean illuminance over vertical planes'), ('Panoramic illuminance (Scalar)', scalar, 'lx', True, 'Scalar = (1/4) integral over the sphere of L(omega) dOmega; mean spherical illuminance'), ('Luminance coefficient of variation (CV_L)', cv_l, 'Dimensionless', False, 'Whole-sphere solid-angle-weighted luminance standard deviation divided by mean luminance'), ('Illuminance coefficient of variation (CV_E)', cv_e, 'Dimensionless', False, 'Standard deviation of directional illuminance divided by its mean')]
    for (angle, value) in zip(range(0, 360, 45), vertical):
        rows.append(('Vertical illuminance %d degrees' % angle, value, 'lx', True, '0 degrees is at the panorama left/right boundary; 180 degrees is at its center'))
    rows.extend([('Diffuseness', diffuseness, 'Dimensionless', False, 'First-order spherical harmonic diffuseness; invariant under uniform photometric scaling'), ('Mean luminance of the 40-degree horizontal band', band_l, 'cd/m²', True, 'Solid-angle-weighted mean luminance in the band from 70 to 110 degrees zenith angle')])
    if any((not np.isfinite(float(row[1])) for row in rows)):
        raise ValueError('Invalid results were found; check the HDR content and projection')
    return {'source': str(path), 'shape_hwc': list(shape), 'rows': [(name, float(value), unit, scaled, definition) for (name, value, unit, scaled, definition) in rows]}


def _parse_positive_number(value,label):
    """Parse a complete value; partial input is allowed while editing."""
    import unicodedata
    text=unicodedata.normalize('NFKC',str(value)).strip().replace('。','.')
    # Accept a single decimal comma; mixed separators/thousands are ambiguous.
    if ',' in text:
        if '.' in text or text.count(',')!=1:
            raise ValueError('Enter a number without units or thousands separators.')
        text=text.replace(',','.')
    try:
        number=float(text)
    except (TypeError,ValueError):
        raise ValueError('Enter a valid value for %s.' % label) from None
    if not np.isfinite(number) or number<=0:
        raise ValueError('%s must be a finite number greater than zero.' % label)
    return number


def parse_measured_illuminance(value):
    return _parse_positive_number(value,'Measured panoramic illuminance (lx)')


def parse_calibration_factor(value):
    return _parse_positive_number(value,'Calibration factor k')


def _numeric_entry_text(text):
    """One-to-one substitutions keep the caret and selection in place."""
    source='０１２３４５６７８９。．｡，,'
    target='0123456789.....'
    return text.translate(str.maketrans(source,target))


def _numeric_entry_ime(entry):
    """Detach IME from this numeric dialog only; return a restoration callback."""
    import sys
    if sys.platform!='win32':
        return lambda:None
    import ctypes
    from ctypes import wintypes
    saved=[]
    associate=None
    def restore():
        if associate is not None:
            while saved:
                hwnd,context=saved.pop()
                associate(hwnd,context)
    try:
        associate=ctypes.WinDLL('imm32').ImmAssociateContext
        associate.argtypes=(wintypes.HWND,wintypes.HANDLE)
        associate.restype=wintypes.HANDLE
        ancestor=ctypes.WinDLL('user32').GetAncestor
        ancestor.argtypes=(wintypes.HWND,wintypes.UINT)
        ancestor.restype=wintypes.HWND
        # Tk routes native focus/composition through its toplevel window.
        toplevel=entry.winfo_toplevel().winfo_id()
        handles=(entry.winfo_id(),toplevel,ancestor(toplevel,2))  # GA_ROOT
        for hwnd in dict.fromkeys(handles):
            if hwnd:
                saved.append((hwnd,associate(hwnd,None)))
    except (OSError,AttributeError):
        restore()
    return restore


def _bind_numeric_entry(entry,variable):
    """Normalize live text and handle one decimal-key press once."""
    import sys
    pending=None
    updating=False
    def normalize():
        nonlocal pending,updating
        pending=None
        original=variable.get()
        normalized=_numeric_entry_text(original)
        if normalized==original:
            return
        caret=entry.index('insert')
        selection=(entry.index('sel.first'),entry.index('sel.last')) if entry.selection_present() else None
        updating=True
        try:
            variable.set(normalized)
            entry.icursor(caret)
            if selection:
                entry.selection_range(*selection)
        finally:
            updating=False
    def changed(*_args):
        nonlocal pending
        if not updating and pending is None:
            # Wait until Tk finishes the edit; do not reset its insertion cursor.
            pending=entry.after_idle(normalize)
    def decimal_key(event):
        if event.state & (0x4|0x8|0x20000):  # Keep Ctrl/Alt shortcuts intact.
            return
        is_decimal=(event.keysym in ('period','KP_Decimal')
                    or event.char in ('.','。','．','｡')
                    or (sys.platform=='win32' and event.keycode in (190,110)))
        if is_decimal:
            if entry.selection_present():
                entry.delete('sel.first','sel.last')
            entry.insert('insert','.')
            return 'break'
    trace=variable.trace_add('write',changed)
    binding=entry.bind('<KeyPress>',decimal_key,add='+')
    ime_restorers=[_numeric_entry_ime(entry)]
    def mapped(_event):
        # The native wrapper may be created after Dialog.body() returns.
        ime_restorers.append(_numeric_entry_ime(entry))
    map_binding=entry.bind('<Map>',mapped,add='+')
    def cleanup():
        if pending is not None:
            entry.after_cancel(pending)
        variable.trace_remove('write',trace)
        entry.unbind('<KeyPress>',binding)
        entry.unbind('<Map>',map_binding)
        while ime_restorers:
            ime_restorers.pop()()
    return cleanup


def ask_calibration(parent):
    import tkinter as tk
    from tkinter import messagebox,simpledialog
    class CalibrationDialog(simpledialog.Dialog):
        def body(self,master):
            self.cleanups=[]
            tk.Label(master,text='Select one calibration method and enter its value.',anchor='w',
                     font=('Segoe UI',10)).grid(row=0,column=0,columnspan=2,sticky='w',pady=(4,10))
            self.mode=tk.StringVar(master=self,value='measured_e')
            self.values={}
            self.entries={}
            for row,(mode,label) in enumerate([
                    ('measured_e','Measured panoramic illuminance (lx)'),
                    ('factor','Known calibration factor k')],1):
                tk.Radiobutton(master,text=label,variable=self.mode,value=mode,
                               command=self.select_mode,font=('Segoe UI',10)).grid(
                                   row=row,column=0,sticky='w',padx=(0,12),pady=6)
                value=tk.StringVar(master=self,value='')
                entry=tk.Entry(master,textvariable=value,width=23,font=('Segoe UI',12))
                entry.grid(row=row,column=1,sticky='ew',pady=6)
                self.values[mode]=value
                self.entries[mode]=entry
                self.cleanups.append(_bind_numeric_entry(entry,value))
            self.select_mode()
            return self.entries[self.mode.get()]
        def select_mode(self):
            selected=self.mode.get()
            for mode,entry in self.entries.items():
                entry.configure(state='normal' if mode==selected else 'disabled')
            self.initial_focus=self.entries[selected]
            self.initial_focus.focus_set()
        def buttonbox(self):
            box=tk.Frame(self)
            tk.Button(box,text='OK',width=10,command=self.ok,default='active').pack(side='left',padx=8,pady=6)
            tk.Button(box,text='Cancel',width=10,command=self.cancel).pack(side='left',padx=8,pady=6)
            box.pack(pady=(0,6))
            self.bind('<Return>',self.ok)
            self.bind('<Escape>',self.cancel)
        def validate(self):
            mode=self.mode.get()
            try:
                if mode=='factor':
                    self.result={'calibration_factor':parse_calibration_factor(self.values[mode].get())}
                else:
                    self.result={'measured_illuminance':parse_measured_illuminance(self.values[mode].get())}
            except ValueError as exc:
                messagebox.showerror('Invalid input',str(exc),parent=self)
                self.entries[mode].selection_range(0,'end')
                return False
            return True
        def destroy(self):
            while getattr(self,'cleanups',[]):
                self.cleanups.pop()()
            super().destroy()
    return CalibrationDialog(parent,'Photometric Calibration').result


PARAMETER_DEFINITIONS={
    'Diffuseness':'Degree of light-field diffuseness, calculated from the zeroth- and first-order spherical harmonic components.',
    'Illuminance coefficient of variation':'Standard deviation of directional illuminance divided by its mean.',
    'Panoramic illuminance':'Measured panoramic illuminance at the measurement position.',
    'Horizontal illuminance':'Illuminance received by a horizontal plane with its normal pointing toward the zenith.',
    'Cylindrical illuminance':'Circumferential mean illuminance over vertical planes in all horizontal directions.',
    'Vertical illuminance':'Illuminance on a vertical plane facing 180 degrees; see the direction sheet for other orientations.',
    'Luminance coefficient of variation':'Whole-sphere luminance standard deviation divided by its mean, weighted by pixel solid angle.',
    '40-degree horizontal band luminance':'Solid-angle-weighted mean luminance in the horizontal band from 70 to 110 degrees zenith angle.',
}
DIRECTION_DEFINITION='Illuminance on a vertical plane at the stated orientation; 0 degrees is the panorama boundary and 180 degrees is the center.'


def calculate_metrics(input_path, measured_illuminance=None, progress=None, *,calibration_factor=None):
    """Use exactly one calibration input and apply k inside the photometric algorithms."""
    if (measured_illuminance is None)==(calibration_factor is None):
        raise ValueError('Provide exactly one of measured panoramic illuminance or calibration factor k.')
    measured=parse_measured_illuminance(measured_illuminance) if measured_illuminance is not None else None
    factor=parse_calibration_factor(calibration_factor) if calibration_factor is not None else None
    mode='measured_e' if measured is not None else 'factor'
    progress=progress or (lambda message: print(message,flush=True))
    path=Path(input_path).expanduser().resolve()
    try:
        raw_e=None
        if mode=='measured_e':
            progress('Deriving k from measured panoramic illuminance and the current HDR...')
            factor,raw_e=calculate_panorama_calibration_factor(str(path),measured)
        progress('Calculating photometric metrics with k=%.8g...' % factor)
        metrics=_compute_metrics_at_factor(path,factor,progress)
        values=metrics['rows']
        # A measured reference is displayed as entered; with fixed k, report HDR Scalar.
        panoramic_e=measured if mode=='measured_e' else values[3][1]
        panoramic_definition=(PARAMETER_DEFINITIONS['Panoramic illuminance'] if mode=='measured_e'
                               else 'Mean spherical illuminance obtained by integrating HDR luminance after applying calibration factor k.')
        metrics['rows']=[
            ('Diffuseness',values[14][1],'Dimensionless',False,PARAMETER_DEFINITIONS['Diffuseness']),
            ('Illuminance coefficient of variation',values[5][1],'Dimensionless',False,PARAMETER_DEFINITIONS['Illuminance coefficient of variation']),
            ('Panoramic illuminance',panoramic_e,'lx',True,panoramic_definition),
            ('Horizontal illuminance',values[0][1],'lx',True,PARAMETER_DEFINITIONS['Horizontal illuminance']),
            ('Cylindrical illuminance',values[2][1],'lx',True,PARAMETER_DEFINITIONS['Cylindrical illuminance']),
            ('Vertical illuminance',values[1][1],'lx',True,PARAMETER_DEFINITIONS['Vertical illuminance']),
            ('Luminance coefficient of variation',values[4][1],'Dimensionless',False,PARAMETER_DEFINITIONS['Luminance coefficient of variation']),
            ('40-degree horizontal band luminance',values[15][1],'cd/m²',True,PARAMETER_DEFINITIONS['40-degree horizontal band luminance']),
        ]
        metrics['vertical_direction_rows']=[
            ('Vertical illuminance (%d degrees)' % angle,row[1],row[2],True,DIRECTION_DEFINITION)
            for angle,row in zip(range(0,360,45),values[6:14])]
        metrics['measured_illuminance']=measured
        metrics['panoramic_illuminance']=panoramic_e
        metrics['calibration_mode']=mode
        metrics['raw_panoramic_illuminance']=raw_e
        metrics['calibration_factor']=factor
        return metrics
    finally:
        _HDR_CACHE.clear()


def _format_workbook(workbook):
    """Shared readable formatting for every report exported by this script."""
    from openpyxl.styles import Alignment,Border,Font,PatternFill,Side
    center=Alignment(horizontal='center',vertical='center',wrap_text=True)
    thin=Side(style='thin',color='9AADC3')
    strong=Side(style='medium',color='526982')

    def block(sheet,top,bottom,left,right):
        for row in sheet.iter_rows(min_row=top,max_row=bottom,min_col=left,max_col=right):
            for cell in row:
                cell.border=Border(
                    left=strong if cell.column==left else thin,
                    right=strong if cell.column==right else thin,
                    top=strong if cell.row==top else thin,
                    bottom=strong if cell.row==bottom else thin)

    for sheet in workbook:
        for row in sheet:
            for cell in row:
                cell.alignment=center
                cell.font=Font(name='Calibri',size=11,color='23364D')
        sheet.sheet_view.showGridLines=False

    for name,header_row,first_row,title_height in [
        ('Metrics Summary',7,8,42),('Vertical by Direction',4,5,38)]:
        sheet=workbook[name]
        sheet.sheet_properties.tabColor='17365D' if name=='Metrics Summary' else '527CAD'
        sheet['A1'].font=Font(name='Calibri',size=16,bold=True,color='FFFFFF')
        sheet.row_dimensions[1].height=title_height
        block(sheet,1,1,1,4)
        if name=='Vertical by Direction':
            sheet.row_dimensions[2].height=36
            block(sheet,2,2,1,4)
            sheet['A2'].fill=PatternFill('solid',fgColor='F1F5FA')
        for column,width in zip('ABCD',(32,20,14,86)):
            sheet.column_dimensions[column].width=width
        for row_number in range(first_row,sheet.max_row+1):
            sheet.row_dimensions[row_number].height=40
            for cell in sheet[row_number]:
                cell.fill=PatternFill('solid',fgColor='EDF3F8' if row_number%2 else 'FFFFFF')
            sheet.cell(row_number,1).font=Font(name='Calibri',size=11,bold=True,color='23364D')
            sheet.cell(row_number,2).font=Font(name='Calibri',size=12,bold=True,color='17365D')
            sheet.cell(row_number,2).fill=PatternFill('solid',fgColor='DDEBF7')
        block(sheet,header_row,sheet.max_row,1,4)
        for cell in sheet[header_row]:
            cell.font=Font(name='Calibri',size=11,bold=True,color='FFFFFF')
            cell.border=Border(left=thin,right=thin,top=strong,bottom=strong)
        sheet.cell(header_row,1).border=Border(left=strong,right=thin,top=strong,bottom=strong)
        sheet.cell(header_row,4).border=Border(left=thin,right=strong,top=strong,bottom=strong)
        sheet.row_dimensions[header_row].height=32

    summary=workbook['Metrics Summary']
    block(summary,2,5,1,4)
    for row_number in range(2,6):
        summary.row_dimensions[row_number].height=34 if row_number!=4 else 40
        summary.cell(row_number,1).fill=PatternFill('solid',fgColor='E2EAF3')
        summary.cell(row_number,1).font=Font(name='Calibri',size=11,bold=True,color='17365D')
    for address in ('B4','D4'):
        summary[address].font=Font(name='Calibri',size=12,bold=True,color='17365D')
        summary[address].fill=PatternFill('solid',fgColor='DDEBF7')
    summary['C4'].font=Font(name='Calibri',size=11,bold=True,color='17365D')
    summary['C4'].fill=PatternFill('solid',fgColor='E2EAF3')
    summary.row_dimensions[6].height=10

    notes=workbook['Calculation Notes']
    notes.sheet_properties.tabColor='8FA3B4'
    notes.column_dimensions['A'].width=24
    notes.column_dimensions['B'].width=112
    block(notes,1,notes.max_row,1,2)
    for cell in notes[1]:
        cell.font=Font(name='Calibri',size=11,bold=True,color='FFFFFF')
    for row_number in range(2,notes.max_row+1):
        notes.row_dimensions[row_number].height=44
        for cell in notes[row_number]:
            cell.fill=PatternFill('solid',fgColor='EDF3F8' if row_number%2 else 'FFFFFF')
        notes.cell(row_number,1).font=Font(name='Calibri',size=11,bold=True,color='17365D')


def export_excel(metrics, output_path):
    from datetime import datetime
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    factor=float(metrics['calibration_factor'])
    if not np.isfinite(factor) or factor<=0:
        raise ValueError('Invalid calibration factor')
    fixed_factor=metrics.get('calibration_mode','measured_e')=='factor'
    path=Path(output_path).expanduser().resolve()
    if path.suffix.lower()!='.xlsx':
        raise ValueError('The output file must use the .xlsx extension')
    workbook=Workbook()
    sheet=workbook.active
    sheet.title='Metrics Summary'
    sheet.merge_cells('A1:D1')
    sheet['A1']='Basic Photometric Metrics'
    sheet['A1'].font=Font(name='Calibri',size=18,bold=True,color='FFFFFF')
    sheet['A1'].fill=PatternFill('solid',fgColor='17365D')
    sheet.row_dimensions[1].height=36
    for row,label,value in [
        (2,'Input HDR',metrics['source']),
        (3,'Projection and size','2:1 equirectangular; %d x %d' % (metrics['shape_hwc'][1],metrics['shape_hwc'][0])),
        (5,'Exported at',datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    ]:
        sheet.cell(row,1,label)
        sheet.merge_cells(start_row=row,start_column=2,end_row=row,end_column=4)
        sheet.cell(row,2,value).alignment=Alignment(wrap_text=True,vertical='center')
    sheet.row_dimensions[2].height=32
    sheet['A4']='Panoramic illuminance (lx, calculated)' if fixed_factor else 'Measured panoramic illuminance (lx, input)'
    sheet['B4']=metrics['panoramic_illuminance']
    sheet['B4'].number_format='0.000'
    sheet['C4']='Calibration factor k (input)' if fixed_factor else 'Calibration factor k (calculated)'
    sheet['D4']=factor
    sheet['D4'].number_format='0.000000'
    sheet['D4'].font=Font(bold=True)
    sheet['D4'].fill=PatternFill('solid',fgColor='E2EFDA')
    sheet.row_dimensions[4].height=28
    headers=['Parameter','Final value','Unit','Definition and notes']
    for column,title in enumerate(headers,1):
        cell=sheet.cell(7,column,title)
        cell.fill=PatternFill('solid',fgColor='24476B')
        cell.font=Font(bold=True,color='FFFFFF')
        cell.alignment=Alignment(vertical='center',wrap_text=True)
    sheet.row_dimensions[7].height=28
    for row_number,(name,value,unit,_scaled,definition) in enumerate(metrics['rows'],8):
        for column,content in enumerate((name,value,unit,definition),1):
            cell=sheet.cell(row_number,column,content)
            cell.fill=PatternFill('solid',fgColor='EDF3F8' if row_number%2 else 'FFFFFF')
            cell.alignment=Alignment(vertical='center',wrap_text=True)
            cell.border=Border(bottom=Side(style='hair',color='D9E2F3'))
        sheet.cell(row_number,2).number_format='0.000'
        sheet.row_dimensions[row_number].height=34
    for column,width in zip('ABCD',(39,22,14,95)):
        sheet.column_dimensions[column].width=width
    sheet.freeze_panes='B8'
    sheet.auto_filter.ref='A7:D%d' % sheet.max_row
    sheet.sheet_view.showGridLines=False
    directions=workbook.create_sheet('Vertical by Direction')
    directions.merge_cells('A1:D1')
    directions['A1']='Vertical Illuminance by Direction'
    directions['A1'].font=Font(name='Calibri',size=16,bold=True,color='FFFFFF')
    directions['A1'].fill=PatternFill('solid',fgColor='17365D')
    directions.row_dimensions[1].height=32
    directions.merge_cells('A2:D2')
    directions['A2']='All values include calibration. 0 degrees is the panorama boundary and 180 degrees is its center. The summary uses 180 degrees.'
    directions['A2'].alignment=Alignment(wrap_text=True,vertical='center')
    directions.row_dimensions[2].height=32
    for column,title in enumerate(headers,1):
        cell=directions.cell(4,column,title)
        cell.fill=PatternFill('solid',fgColor='24476B')
        cell.font=Font(bold=True,color='FFFFFF')
        cell.alignment=Alignment(wrap_text=True,vertical='center')
    for row_number,(name,value,unit,_scaled,definition) in enumerate(metrics['vertical_direction_rows'],5):
        for column,content in enumerate((name,value,unit,definition),1):
            cell=directions.cell(row_number,column,content)
            cell.fill=PatternFill('solid',fgColor='EDF3F8' if row_number%2 else 'FFFFFF')
            cell.alignment=Alignment(wrap_text=True,vertical='center')
        directions.cell(row_number,2).number_format='0.000'
        directions.row_dimensions[row_number].height=32
    for column,width in zip('ABCD',(39,22,14,95)):
        directions.column_dimensions[column].width=width
    directions.freeze_panes='B5'
    directions.auto_filter.ref='A4:D12'
    directions.sheet_view.showGridLines=False
    notes=workbook.create_sheet('Calculation Notes')
    notes.append(['Item','Description'])
    calibration_note=('Uses the entered calibration factor k directly; measured panoramic illuminance is not required.' if fixed_factor
                      else 'Derives k from the measured reference: k = measured panoramic illuminance / uncalibrated HDR panoramic illuminance.')
    result_note=('Metrics Summary D4 records the entered k. B4 and panoramic illuminance are HDR integration results. Column B of the metric table contains final values.' if fixed_factor
                 else 'Metrics Summary B4 records the measured input; D4 records the derived k. Column B of the metric table contains final values.')
    panorama_note=('After applying k, mean spherical illuminance is calculated as Scalar = (1/4) integral over the sphere of L dOmega.' if fixed_factor
                   else 'Records the measured panoramic illuminance; HDR integration is used to derive k.')
    for row in [
        ('Input requirements','Use a stitched 360 x 180 degree, 2:1 equirectangular HDR with the zenith at the top. Stitch stacked fisheyes first.'),
        ('Calibration input',calibration_note),
        ('Result locations',result_note),
        ('Final results','Results include calibration factor k. Rerun the tool after changing the calibration input; this workbook is a calculation snapshot.'),
        ('Display precision','Metrics display three decimal places and k displays six; numeric cells retain full calculation precision.'),
        ('Panoramic illuminance',panorama_note),
        ('Reported metrics','The summary contains eight basic metrics; vertical illuminance at eight orientations is listed separately.'),
        ('RGB to luminance','L = 179 x (0.2126R + 0.7152G + 0.0722B) x k. The constant 179 does not replace measured photometric calibration.'),
        ('Coefficient of variation','CV is standard deviation divided by the mean. Luminance CV uses pixel solid-angle weights; illuminance CV uses directional samples.'),
        ('Vertical orientation','0 degrees is the panorama boundary and 180 degrees is its center. Changing the panorama yaw changes the corresponding vertical illuminance.'),
        ('Scalar convention','Scalar = (1/4) integral over the sphere of L dOmega; it represents mean spherical illuminance.'),
        ('Sampling','Calibration, Scalar and spherical harmonics use 5% nearest-neighbor sampling; directional illuminance and CV use 2.5%. Luminance CV uses exact solid angles; the 40-degree band uses the full image.'),
        ('Dimensionless metrics','Coefficients of variation and diffuseness are invariant under uniform photometric scaling; do not multiply them separately by k.'),
        ('Photometric Calibration','Illuminance and luminance from an uncalibrated HDR require factor k. This tool applies k during calculation; do not apply it again. Vignetting correction is not photometric calibration.'),
        ('Calibration applicability','The measured reference or known k must apply to the current HDR and its acquisition and processing conditions. Confirm calibration when changing HDRs.'),
    ]:
        notes.append(row)
    notes.column_dimensions['A'].width=23
    notes.column_dimensions['B'].width=122
    for row in notes:
        for cell in row:cell.alignment=Alignment(wrap_text=True,vertical='center')
        notes.row_dimensions[row[0].row].height=36
    for cell in notes[1]:
        cell.fill=PatternFill('solid',fgColor='24476B')
        cell.font=Font(bold=True,color='FFFFFF')
    notes.freeze_panes='B2'
    notes.sheet_view.showGridLines=False
    _format_workbook(workbook)
    path.parent.mkdir(parents=True,exist_ok=True)
    workbook.save(path)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',nargs='?',help='A stitched 2:1 equirectangular panorama HDR')
    parser.add_argument('-o','--output',help='Full output path for the Excel workbook (.xlsx)')
    calibration=parser.add_mutually_exclusive_group()
    calibration.add_argument('--measured-e',type=float,help='Measured panoramic illuminance in lx; used to calculate k')
    calibration.add_argument('--factor','--k',dest='factor',type=float,help='Known calibration factor k; no measured illuminance required')
    args=parser.parse_args()
    if args.measured_e is not None:
        try:parse_measured_illuminance(args.measured_e)
        except ValueError as exc:parser.error(str(exc))
        calibration_input={'measured_illuminance':args.measured_e}
    elif args.factor is not None:
        try:parse_calibration_factor(args.factor)
        except ValueError as exc:parser.error(str(exc))
        calibration_input={'calibration_factor':args.factor}
    else:
        calibration_input=None
    if args.input and args.output and calibration_input is not None:
        result=export_excel(calculate_metrics(args.input,**calibration_input),args.output)
        print('Saved: %s' % result,flush=True)
        return
    import tkinter as tk
    from tkinter import filedialog,messagebox
    root=tk.Tk();root.withdraw()
    try:
        source=args.input or filedialog.askopenfilename(
            parent=root,title='Select a 2:1 equirectangular panorama HDR',filetypes=[('HDR files','*.hdr')])
        if not source:return
        if calibration_input is None:
            calibration_input=ask_calibration(root)
            if calibration_input is None:return
        target=args.output or filedialog.asksaveasfilename(
            parent=root,title='Save the photometric metrics workbook',initialdir=str(Path(source).parent),
            initialfile=Path(source).stem+'_metrics.xlsx',defaultextension='.xlsx',
            filetypes=[('Excel workbook','*.xlsx')],confirmoverwrite=True)
        if not target:return
        root.title('HDR Photometric Metrics');root.geometry('520x130');root.resizable(False,False)
        root.protocol('WM_DELETE_WINDOW',lambda:None)
        status=tk.StringVar(value='Preparing...')
        tk.Label(root,textvariable=status,padx=20,pady=25,wraplength=480).pack();root.deiconify()
        def progress(message):
            status.set(message);root.update()
        try:
            metrics=calculate_metrics(source,progress=progress,**calibration_input)
            result=export_excel(metrics,target)
        except Exception as exc:
            messagebox.showerror('Calculation failed',str(exc),parent=root);return
        calibration_label='Entered k' if metrics['calibration_mode']=='factor' else 'Calculated k'
        panorama_label='Calculated panoramic illuminance' if metrics['calibration_mode']=='factor' else 'Measured panoramic illuminance'
        messagebox.showinfo(
            'Calculation complete','Excel workbook saved:\n%s\n\n%s: %.8g lx\n%s: %.8g\nPhotometric calibration has been applied.'
            % (result,panorama_label,metrics['panoramic_illuminance'],calibration_label,metrics['calibration_factor']),parent=root)
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
