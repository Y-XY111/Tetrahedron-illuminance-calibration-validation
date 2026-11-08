from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def read_stacked_hdr(input_path):
    """Decode once; split and rotate the two fisheyes entirely in memory."""
    source=Path(input_path)
    hdr=cv2.imdecode(np.fromfile(str(source), np.uint8), cv2.IMREAD_UNCHANGED)
    if hdr is None:
        raise ValueError('Cannot read HDR: %s' % source)
    if hdr.ndim != 3 or hdr.shape[2] != 3:
        raise ValueError('A three-channel dual-fisheye HDR is required')
    if not np.issubdtype(hdr.dtype, np.floating) or not np.all(np.isfinite(hdr)):
        raise ValueError('HDR pixels must contain finite floating-point values')
    height, width=hdr.shape[:2]
    if height != 2 * width:
        raise ValueError('The input must contain two square fisheyes stacked vertically; height must equal twice the width')
    return [cv2.rotate(hdr[:width], cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(hdr[width:], cv2.ROTATE_90_COUNTERCLOCKWISE)]


def feature_view(hdr):
    gray = cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)
    positive = gray[gray > 0]
    middle = np.median(positive)
    scaled = gray / max(float(middle), 1e-10) * 0.18
    gray = np.round((scaled / (1 + scaled)) ** (1 / 2.2) * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2, tileGridSize=(12, 12)).apply(gray)


def project(directions, fov, shape, center=None):
    h, w = shape[:2]
    cx, cy = center if center is not None else ((w - 1) / 2, (h - 1) / 2)
    theta = np.arctan2(np.linalg.norm(directions[..., :2], axis=-1), directions[..., 2])
    angle = np.arctan2(directions[..., 1], directions[..., 0])
    r = theta / np.deg2rad(fov / 2) * min(h, w) / 2
    return (cx + r * np.cos(angle)).astype(np.float32), (cy - r * np.sin(angle)).astype(np.float32), r


def ring_maps(fov, shape, width=4096, height=512, band_degrees=22):
    t = (np.arange(width, dtype=np.float32) + .5) / width * (2 * np.pi)
    d = (np.arange(height, dtype=np.float32) + .5 - height / 2) / height * np.deg2rad(2 * band_degrees)
    tt, dd = np.meshgrid(t, d)
    directions = np.stack([np.cos(dd) * np.sin(tt), np.cos(dd) * np.cos(tt), np.sin(dd)], axis=-1)
    first = project(directions, fov, shape)
    second = project(directions @ np.diag([-1, 1, -1]), fov, shape)
    return first, second


def match_rings(images, fov, features=None):
    maps = ring_maps(fov, images[0].shape)
    if features is None:
        features = [feature_view(x) for x in images]
    points, descriptors = [], []
    detector = cv2.SIFT_create(nfeatures=10000, contrastThreshold=.012, edgeThreshold=12)
    for feature, (xmap, ymap, radius) in zip(features, maps):
        ring = cv2.remap(feature, xmap, ymap, cv2.INTER_LINEAR)
        mask = ((radius > feature.shape[0] * .38) & (radius < feature.shape[0] * .485)).astype(np.uint8) * 255
        kp, desc = detector.detectAndCompute(ring, mask)
        points.append(kp)
        descriptors.append(desc)
    print('fov', fov, 'keypoints', [len(p) for p in points], flush=True)
    if any(d is None for d in descriptors):
        return np.empty((0, 2, 2))
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(descriptors[0], descriptors[1], k=2)
    reverse = matcher.knnMatch(descriptors[1], descriptors[0], k=2)
    reverse_good = {m.queryIdx: m.trainIdx for pair in reverse if len(pair) == 2 for m, n in [pair] if m.distance < .78 * n.distance}
    good = [m for pair in forward if len(pair) == 2 for m, n in [pair] if m.distance < .78 * n.distance and reverse_good.get(m.trainIdx) == m.queryIdx]
    measurements = []
    for m in good:
        pair = []
        for index, keypoint in enumerate((points[0][m.queryIdx], points[1][m.trainIdx])):
            x, y = keypoint.pt
            xmap, ymap, _ = maps[index]
            xy = np.array([[[x, y]]], np.float32)
            sx = cv2.remap(xmap, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR).item()
            sy = cv2.remap(ymap, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR).item()
            pair.append([sx, sy])
        measurements.append(pair)
    print('mutual matches', len(measurements), flush=True)
    return np.asarray(measurements)


def unproject(points, fov, center, radius):
    x = (points[:, 0] - center[0]) / radius
    y = (center[1] - points[:, 1]) / radius
    rr = np.hypot(x, y)
    theta = rr * np.deg2rad(fov / 2)
    angle = np.arctan2(y, x)
    return np.column_stack([np.sin(theta) * np.cos(angle), np.sin(theta) * np.sin(angle), np.cos(theta)])


def fit_matches(matches, shape):
    _, indices = np.unique(np.round(matches.reshape(-1, 4) / 8), axis=0, return_index=True)
    matches = matches[np.sort(indices)]
    h, w = shape[:2]
    center, radius = ((w - 1) / 2, (h - 1) / 2), min(h, w) / 2
    base = np.diag([-1., 1., -1.])
    xy = matches.copy()
    xy[:, :, 0] -= center[0]
    xy[:, :, 1] = center[1] - xy[:, :, 1]
    az = np.arctan2(xy[:, :, 1], xy[:, :, 0])
    az_error = np.abs(np.arctan2(np.sin(az[:, 0] + az[:, 1] - np.pi), np.cos(az[:, 0] + az[:, 1] - np.pi)))

    def residual(p, measurements):
        rot = Rotation.from_rotvec(p[1:]).as_matrix() @ base
        first = unproject(measurements[:, 0], p[0], center, radius)
        second = unproject(measurements[:, 1], p[0], center, radius)
        return (first @ rot.T - second).ravel()

    keep = az_error < np.deg2rad(12)
    fit = least_squares(residual, [195., 0, 0, 0], args=(matches[keep],),
                        bounds=([175, -.3, -.3, -.3], [235, .3, .3, .3]),
                        loss='soft_l1', f_scale=.012, max_nfev=200)
    for _ in range(4):
        errors = np.linalg.norm(residual(fit.x, matches).reshape(-1, 3), axis=1)
        keep = errors < np.deg2rad(1.5)
        if np.count_nonzero(keep) < 12:
            raise RuntimeError('Not enough geometrically consistent matches')
        fit = least_squares(residual, fit.x, args=(matches[keep],),
                            bounds=([175, -.3, -.3, -.3], [235, .3, .3, .3]),
                            loss='soft_l1', f_scale=.006, max_nfev=200)
    errors = np.linalg.norm(residual(fit.x, matches).reshape(-1, 3), axis=1)
    keep = errors < np.deg2rad(1.5)
    rotation = Rotation.from_rotvec(fit.x[1:]).as_matrix() @ base
    info = {'fov_degrees': float(fit.x[0]), 'rotation_first_to_second': rotation.tolist(),
            'inlier_count': int(keep.sum()), 'candidate_count': len(matches),
            'median_angular_error_degrees': float(np.rad2deg(np.median(errors[keep]))),
            'p90_angular_error_degrees': float(np.rad2deg(np.percentile(errors[keep], 90))),
            'inlier_source_points_after_rotation': matches[keep].tolist(),
            'model': 'shared equidistant fisheye, centered image circles, relative 3D rotation',
            'source_shape_hwc': list(shape),
            'upright_rotations': {'upper': '90 degrees clockwise', 'lower': '90 degrees counterclockwise'}}
    return info


def choose_seam(images, info, width=4096, height=257):
    """Find a closed low-difference seam inside the spherical overlap band."""
    fov = info['fov_degrees']
    rotation = np.asarray(info['rotation_first_to_second'])
    limit = min(8., (fov / 2 * .965 - 90) * .8)
    if limit <= .5:
        raise RuntimeError('Too little overlap for a reliable seam')
    azimuth = np.arange(width, dtype=np.float32) / width * (2 * np.pi)
    delta = np.linspace(-limit, limit, height, dtype=np.float32) * np.pi / 180
    tt, dd = np.meshgrid(azimuth, delta)
    directions = np.stack([np.cos(dd) * np.sin(tt), np.cos(dd) * np.cos(tt), np.sin(dd)], axis=-1)
    rings, valid = [], []
    for index, hdr in enumerate(images):
        rays = directions if index == 0 else directions @ rotation.T
        xmap, ymap, radius = project(rays, fov, hdr.shape)
        gray = cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)
        scale = max(float(np.median(gray[gray > 0])), 1e-10)
        ring = cv2.remap(gray, xmap, ymap, cv2.INTER_LINEAR)
        rings.append(np.log1p(ring / scale))
        valid.append(radius < min(hdr.shape[:2]) / 2 * .965)
    difference = np.abs(rings[0] - rings[1])
    gradients = []
    for ring in rings:
        gradients.append(cv2.Sobel(ring, cv2.CV_32F, 1, 0, ksize=3) / 8)
    cost = cv2.GaussianBlur(difference + .5 * np.abs(gradients[0] - gradients[1]), (5, 5), 0)
    cost += .03 * (dd / np.deg2rad(limit)) ** 2
    cost[~(valid[0] & valid[1])] = 1e4
    start = int(np.argmin(np.mean(cost[:, :8], axis=1) + np.mean(cost[:, -8:], axis=1)))
    previous = np.full(height, 1e9, np.float32)
    previous[start] = cost[start, 0]
    parents = np.zeros((width, height), np.int16)
    indices = np.arange(height)
    for column in range(1, width):
        shifted = np.stack([np.r_[1e9, previous[:-1]] + .002, previous, np.r_[previous[1:], 1e9] + .002])
        selected = np.argmin(shifted, axis=0)
        parents[column] = indices + selected - 1
        previous = np.min(shifted, axis=0) + cost[:, column]
    path = np.empty(width, np.int32)
    path[-1] = start
    for column in range(width - 1, 0, -1):
        path[column - 1] = parents[column, path[column]]
    seam = delta[path]
    seam = (np.roll(seam, 2) + 2 * np.roll(seam, 1) + 3 * seam + 2 * np.roll(seam, -1) + np.roll(seam, -2)) / 9
    return seam


def render(images, info, width=4096, seam=None, progress=None):
    height = width // 2
    result = np.zeros((height, width, 3), np.float32)
    fov = info['fov_degrees']
    rotation = np.asarray(info['rotation_first_to_second'])
    longitude = (np.arange(width, dtype=np.float32) + .5) / width * (2 * np.pi) - np.pi
    theta_max = np.deg2rad(fov / 2) * .965
    for top in range(0, height, 128):
        bottom = min(top + 128, height)
        latitude = np.pi / 2 - (np.arange(top, bottom, dtype=np.float32) + .5) / height * np.pi
        lon, lat = np.meshgrid(longitude, latitude)
        directions = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1)
        first_weights = None
        if seam is not None:
            azimuth = np.mod(np.arctan2(directions[..., 0], directions[..., 1]), 2 * np.pi)
            azimuth_indices = azimuth / (2 * np.pi) * len(seam)
            seam_values = np.interp(azimuth_indices, np.arange(len(seam) + 1), np.r_[seam, seam[0]])
            delta = np.arcsin(np.clip(directions[..., 2], -1, 1))
            # Only the narrow seam is feathered; HDR radiometric gains stay at 1.
            blend = np.clip((delta - seam_values) / np.deg2rad(.65) + .5, 0, 1)
            first_weights = (blend * blend * (3 - 2 * blend)).astype(np.float32)
        numerator = np.zeros_like(directions, dtype=np.float32)
        denominator = np.zeros(directions.shape[:2], np.float32)
        for index, hdr in enumerate(images):
            rays = directions if index == 0 else directions @ rotation.T
            xmap, ymap, radius = project(rays, fov, hdr.shape)
            values = cv2.remap(hdr, xmap, ymap, cv2.INTER_LINEAR)
            theta = radius / (min(hdr.shape[:2]) / 2) * np.deg2rad(fov / 2)
            if first_weights is None:
                weights = np.clip((theta_max - theta) / np.deg2rad(8), 0, 1)
            else:
                weights = first_weights if index == 0 else 1 - first_weights
                weights = weights * (theta < theta_max)
            numerator += values * weights[..., None]
            denominator += weights
        if np.any(denominator < 1e-8):
            raise RuntimeError('Insufficient source coverage')
        result[top:bottom] = numerator / denominator[..., None]
        if progress is not None and (top % 512 == 0 or bottom == height):
            progress('Rendering panorama: %d%%' % round(bottom / height * 100))
    return result


def stitch(images, width=6000, progress=None):
    if width < 512 or width % 2:
        raise ValueError('Panorama width must be an even integer of at least 512')
    cv2.setNumThreads(2)
    progress = progress or (lambda text: print(text, flush=True))
    progress('Preparing images for feature matching...')
    features = [feature_view(image) for image in images]
    matches = []
    guesses = [185, 190, 195, 200, 205, 210]
    for index, fov in enumerate(guesses, 1):
        progress('Matching fisheye edges: %d/%d' % (index, len(guesses)))
        found = match_rings(images, fov, features)
        if found.size:
            matches.extend(found.tolist())
    if len(matches) < 12:
        raise RuntimeError('Not enough edge matches were found for reliable stitching')
    del features
    progress('Estimating the effective field of view and relative rotation...')
    info = fit_matches(np.asarray(matches), images[0].shape)
    progress('Selecting a seam in the overlap...')
    seam = choose_seam(images, info)
    panorama = render(images, info, width, seam, progress)
    progress('Centering the lower fisheye; placing the upper fisheye across the left and right edges...')
    # A 180-degree yaw is an exact half-width cyclic shift: no resampling or
    # brightness change. Keep the lower fisheye and its board away from the cut.
    panorama = np.roll(panorama, width // 2, axis=1)
    if not np.all(np.isfinite(panorama)) or np.min(panorama) < 0:
        raise RuntimeError('The panorama contains invalid pixel values')
    return panorama


def stacked_hdr_to_panorama(input_path, output_path, width=6000, progress=None):
    """One HDR input and one user-selected HDR output; no intermediate files."""
    if width < 512 or width % 2:
        raise ValueError('Panorama width must be an even integer of at least 512')
    source=Path(input_path).expanduser().resolve()
    target=Path(output_path).expanduser().resolve()
    if target.suffix.lower() != '.hdr':
        raise ValueError('The output file must use the .hdr extension')
    if target == source or (target.exists() and target.samefile(source)):
        raise ValueError('Choose a different output filename to preserve the source HDR')
    progress=progress or (lambda text: print(text, flush=True))
    progress('Reading the HDR, splitting the fisheyes and correcting orientation...')
    images=read_stacked_hdr(source)
    panorama=stitch(images, width, progress)
    progress('Encoding the final panorama HDR...')
    ok,encoded=cv2.imencode('.hdr', panorama)
    if not ok:
        raise RuntimeError('HDR encoding failed')
    decoded=cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape != panorama.shape or not np.all(np.isfinite(decoded)):
        raise RuntimeError('The encoded HDR failed validation')
    del decoded, panorama, images
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(target))
    progress('Completed: %s' % target)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', help='One HDR containing two vertically stacked fisheyes')
    parser.add_argument('-o', '--output', help='Full output path for the panorama HDR')
    parser.add_argument('--width', type=int, default=6000, help='Panorama width in pixels (default: 6000)')
    args=parser.parse_args()
    if args.width < 512 or args.width % 2:
        parser.error('Panorama width must be an even integer of at least 512')
    if args.input and args.output:
        stacked_hdr_to_panorama(args.input, args.output, args.width)
        return

    import tkinter as tk
    from tkinter import filedialog, messagebox

    root=tk.Tk()
    root.withdraw()
    try:
        input_path=args.input or filedialog.askopenfilename(
            parent=root,
            title='Select one vertically stacked dual-fisheye HDR',
            filetypes=[('Radiance HDR files', '*.hdr'), ('All files', '*.*')],
        )
        if not input_path:
            return
        source=Path(input_path).expanduser().resolve()
        output_path=args.output or filedialog.asksaveasfilename(
            parent=root,
            title='Save the final panorama HDR',
            initialdir=str(source.parent),
            initialfile=source.stem + '_panorama.hdr',
            defaultextension='.hdr',
            filetypes=[('Radiance HDR files', '*.hdr')],
            confirmoverwrite=True,
        )
        if not output_path:
            return
        root.title('HDR Panorama Stitching')
        root.geometry('520x140')
        root.resizable(False, False)
        root.protocol('WM_DELETE_WINDOW', lambda: None)
        status=tk.StringVar(value='Preparing to split and stitch...')
        tk.Label(root, textvariable=status, padx=20, pady=30, wraplength=480).pack()
        root.deiconify()

        def progress(message):
            status.set(message)
            root.update()

        try:
            result=stacked_hdr_to_panorama(source, output_path, args.width, progress)
        except Exception as exc:
            messagebox.showerror('Processing failed', str(exc), parent=root)
            return
        messagebox.showinfo('Stitching complete', 'Final panorama HDR saved:\n%s' % result, parent=root)
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
