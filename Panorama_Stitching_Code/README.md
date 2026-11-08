# Panorama Stitching

`hdr_to_panorama.py` splits and stitches one vertically stacked dual-fisheye HDR independently of the main application.

## Run

1. Install Python and the packages in `requirements.txt`.
2. Double-click `run_stitching.bat`, or run `python hdr_to_panorama.py`.
3. Select one source HDR. The example is `Examples/dual_fisheye_example.hdr`.
4. Choose the final HDR filename and output location.
5. Wait for completion. Default output: 6000 x 3000 pixels, 360 x 180 degrees, 2:1 equirectangular projection.

Splitting, orientation correction, matching and stitching happen in memory. Only the final HDR is saved; split images, JPG previews, JSON and validation files are not written. The original lower fisheye is centered; the upper fisheye wraps across the boundaries, keeping the board in the central view.

Command line:

```text
python hdr_to_panorama.py "input.hdr" --output "panorama.hdr" --width 6000
```

The output width must be an even integer of at least 512. The output cannot overwrite the source HDR. Run without arguments to choose both locations in dialogs. Standard file-picker controls follow the operating-system language.

## Supported arrangement

The single input example, `dual_fisheye_example.hdr`, is 3040 x 6080 pixels. Each half is a square fisheye. The upper half is rotated 90 degrees clockwise; the lower half is rotated 90 degrees counterclockwise.

This convention applies to the supplied lens arrangement and is not a universal camera calibration. Effective field of view and relative rotation are estimated from overlapping features. Nearby objects may still show parallax. Output retains linear HDR values through geometric resampling and seam blending. Tone mapping is used only for matching and is not saved in the HDR. The tool uses the CPU without CuPy.

## Photometric calibration

**If the HDR has not been photometrically calibrated, subsequent illuminance and luminance calculations require calibration factor k.** The metrics tool applies either a k derived from measured panoramic illuminance or a known k entered directly. Vignetting correction alone does not establish absolute photometric calibration.
