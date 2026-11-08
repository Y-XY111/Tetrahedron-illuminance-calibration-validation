# Photometric Metrics and Excel Export

`hdr_metrics_to_excel.py` calculates metrics independently of the main application. It does not require Qt, plotting libraries, CuPy, configuration files or application modules.

## Run

1. Install the packages in `requirements.txt`.
2. Double-click `run_metrics.bat`, or run `python hdr_metrics_to_excel.py`.
3. Select one stitched linear HDR: 2:1 equirectangular, covering 360 x 180 degrees, with the zenith at the top. Uncalibrated HDRs are supported. Stitch stacked fisheyes first; do not select JPG/PNG previews or HDR files made from tone-mapped images.
4. In **Photometric Calibration**, choose **Measured panoramic illuminance (lx)** to derive k from a measured reference, or **Known calibration factor k** to enter k directly without measured illuminance.
5. Choose the `.xlsx` output filename and location.
6. Wait for completion. One workbook is saved. Cancelling calibration or saving exits without calculating or exporting.

The unused field is disabled. Switching methods preserves drafts; only the selected field is validated. Values must be finite and greater than zero. Decimal keys insert `.`; full-width digits and decimal punctuation are normalized immediately. Enter numbers without units or thousands separators. A single comma is treated as a decimal separator. Partial values are allowed while typing and checked when you click **OK**. On Windows, IME composition is temporarily disabled only for the numeric dialog and restored when it closes. Standard file-picker controls follow the operating-system language.

## Command line

Derive k from a measured reference:

```text
python hdr_metrics_to_excel.py "panorama.hdr" --output "metrics.xlsx" --measured-e 403.25
```

Use a known coefficient:

```text
python hdr_metrics_to_excel.py "panorama.hdr" --output "metrics.xlsx" --factor 33.40962669
```

`--k` is an alias for `--factor`. `--factor` and `--measured-e` are mutually exclusive. Omitting both opens the calibration window. These numbers are examples; supply inputs applicable to your HDR.

## Calibration and workbook

**Uncalibrated HDR illuminance and luminance require multiplication by k. This tool applies k inside the calculations; do not multiply exported results again.**

Measured-reference mode derives `k = measured panoramic illuminance / uncalibrated HDR panoramic illuminance` and reports the measured input as panoramic illuminance. Known-k mode uses the entered k unchanged and reports panoramic illuminance calculated from the calibrated HDR. A coefficient from another method is supported if it applies to the current HDR and its acquisition and processing conditions. For an already calibrated HDR requiring no further scaling, enter k = 1.

The workbook has three sheets:

- **Metrics Summary:** eight primary metrics. B4 records measured or calculated panoramic illuminance; D4 records derived or entered k. Labels identify each source.
- **Vertical by Direction:** vertical illuminance at 0, 45, 90, 135, 180, 225, 270 and 315 degrees.
- **Calculation Notes:** calibration method, definitions and conventions.

Summary order: Diffuseness, Illuminance coefficient of variation, Panoramic illuminance, Horizontal illuminance, Cylindrical illuminance, Vertical illuminance, Luminance coefficient of variation, and 40-degree horizontal band luminance. The summary uses vertical illuminance at 180 degrees. Zero degrees is the panorama boundary; 180 degrees is its center. Changing panorama yaw changes the corresponding vertical illuminance.

The four table columns are **Parameter**, **Final value**, **Unit** and **Definition and notes**. Raw-value and extra-multiplier columns are omitted. All text and numbers are centered, with clear borders and distinct title, header and data styles. Metrics display three decimals and k displays six; cells retain numeric calculation precision. The workbook is a snapshot: rerun after changing calibration.

## Calculation conventions

- Luminance: `L = 179 x (0.2126R + 0.7152G + 0.0722B) x k`.
- Panoramic illuminance: mean spherical illuminance, `Scalar = (1/4) integral over the sphere of L dOmega`.
- Calibration, Scalar and spherical harmonics use 5% nearest-neighbor sampling. Directional illuminance and CV use 2.5%. Luminance CV uses exact pixel solid-angle weights; band luminance uses the original image.
- The horizontal 40-degree band covers zenith angles from 70 to 110 degrees.
- CV is standard deviation divided by the mean. Luminance CV uses solid-angle weights; illuminance CV uses directional samples. CV and diffuseness receive no additional k multiplier.

The supplied workbooks are measured-reference examples. Their inputs are for demonstration and verification, not defaults for other HDRs.
