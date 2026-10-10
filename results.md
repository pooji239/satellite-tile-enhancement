# Evaluation results

Tiles: 24. Degradation: JPEG q=70, noise sigma=2.0, seed=0. Metrics use roof pixels only (grey background masked out). Higher PSNR/SSIM = closer to the original.

## Experiment A - 2x upscaling (bicubic vs pretrained FSRCNN)

| Tile | Size | Bicubic PSNR | SR PSNR | Bicubic SSIM | SR SSIM |
|---|---|---|---|---|---|
| 1e44ba_google | 320x250 | 25.39 | 25.71 | 0.766 | 0.777 |
| 20d961_google | 944x436 | 25.11 | 25.67 | 0.451 | 0.453 |
| 3d9b33_google | 394x280 | 23.73 | 24.57 | 0.688 | 0.714 |
| 5aca8b_google | 684x594 | 23.19 | 23.38 | 0.563 | 0.569 |
| 666e65_google | 232x224 | 25.33 | 25.45 | 0.678 | 0.684 |
| 686199_google | 258x258 | 24.66 | 24.73 | 0.577 | 0.578 |
| 6c8fd5_google | 422x376 | 28.20 | 28.65 | 0.799 | 0.808 |
| 7345ba_google | 394x472 | 22.20 | 22.24 | 0.511 | 0.512 |
| 761669_google | 262x376 | 24.59 | 24.87 | 0.594 | 0.603 |
| 7ecfdc_google | 226x156 | 24.41 | 24.83 | 0.762 | 0.769 |
| 8899da_google | 124x118 | 28.72 | 30.17 | 0.817 | 0.831 |
| 8d05f2_google | 876x520 | 31.23 | 31.73 | 0.895 | 0.901 |
| 90435e_google | 196x198 | 19.74 | 20.08 | 0.451 | 0.474 |
| 98fa77_google | 200x194 | 24.48 | 24.84 | 0.782 | 0.788 |
| a3a517_google | 952x564 | 22.93 | 23.23 | 0.518 | 0.522 |
| a90908_google | 418x500 | 21.99 | 22.08 | 0.532 | 0.534 |
| aba1eb_google | 96x110 | 27.56 | 28.15 | 0.773 | 0.786 |
| b31756_google | 152x134 | 26.26 | 26.90 | 0.763 | 0.777 |
| c85563_google | 182x302 | 26.74 | 27.72 | 0.838 | 0.852 |
| cac04a_google | 102x74 | 26.53 | 27.90 | 0.809 | 0.834 |
| d87dbc_google | 198x200 | 22.46 | 22.80 | 0.651 | 0.667 |
| e42da1_google | 220x230 | 22.79 | 23.02 | 0.602 | 0.610 |
| f28a5f_google | 630x490 | 25.35 | 26.05 | 0.616 | 0.627 |
| f73de6_google | 620x436 | 29.76 | 30.01 | 0.791 | 0.792 |
| **Mean** | | **25.14** | **25.62** | **0.676** | **0.686** |

SR beats bicubic on PSNR in 24/24 tiles.

## Experiment B - classical pipeline vs doing nothing

`notone` = same pipeline with the CLAHE contrast step switched off (ablation).

| Tile | Degraded PSNR | Classical PSNR | No-tone PSNR | Degraded SSIM | Classical SSIM | No-tone SSIM |
|---|---|---|---|---|---|---|
| 1e44ba_google | 25.97 | 21.91 | 27.07 | 0.803 | 0.806 | 0.841 |
| 20d961_google | 25.63 | 25.49 | 25.88 | 0.478 | 0.419 | 0.410 |
| 3d9b33_google | 24.00 | 20.82 | 25.03 | 0.727 | 0.742 | 0.749 |
| 5aca8b_google | 23.49 | 20.30 | 23.93 | 0.589 | 0.626 | 0.614 |
| 666e65_google | 25.81 | 22.25 | 26.38 | 0.712 | 0.723 | 0.739 |
| 686199_google | 25.14 | 22.48 | 25.63 | 0.611 | 0.645 | 0.632 |
| 6c8fd5_google | 28.90 | 24.90 | 30.19 | 0.848 | 0.833 | 0.855 |
| 7345ba_google | 22.22 | 21.83 | 22.42 | 0.489 | 0.532 | 0.504 |
| 761669_google | 24.97 | 22.90 | 25.51 | 0.637 | 0.634 | 0.634 |
| 7ecfdc_google | 24.62 | 21.01 | 26.39 | 0.790 | 0.792 | 0.840 |
| 8899da_google | 29.28 | 16.19 | 31.54 | 0.881 | 0.719 | 0.885 |
| 8d05f2_google | 31.88 | 30.00 | 32.83 | 0.919 | 0.904 | 0.920 |
| 90435e_google | 19.76 | 18.30 | 20.19 | 0.426 | 0.528 | 0.491 |
| 98fa77_google | 23.93 | 22.55 | 26.01 | 0.807 | 0.832 | 0.857 |
| a3a517_google | 22.83 | 22.51 | 23.47 | 0.530 | 0.556 | 0.539 |
| a90908_google | 21.98 | 21.43 | 22.25 | 0.507 | 0.551 | 0.525 |
| aba1eb_google | 28.45 | 11.48 | 29.65 | 0.828 | 0.630 | 0.840 |
| b31756_google | 26.90 | 23.52 | 28.03 | 0.817 | 0.809 | 0.847 |
| c85563_google | 26.73 | 22.75 | 28.84 | 0.877 | 0.857 | 0.899 |
| cac04a_google | 27.45 | 18.06 | 28.98 | 0.856 | 0.637 | 0.872 |
| d87dbc_google | 22.61 | 20.05 | 23.42 | 0.688 | 0.706 | 0.724 |
| e42da1_google | 23.24 | 21.06 | 23.94 | 0.631 | 0.682 | 0.672 |
| f28a5f_google | 25.58 | 25.03 | 26.09 | 0.666 | 0.624 | 0.620 |
| f73de6_google | 30.60 | 23.91 | 31.79 | 0.839 | 0.820 | 0.857 |
| **Mean** | **25.50** | **21.70** | **26.48** | **0.706** | **0.692** | **0.724** |

Full classical improved PSNR in 0/24 tiles (hurt in 24).

No-tone variant improved PSNR in 24/24 tiles (hurt in 0).

Caveat: the degradation includes a blur, which favours sharpening; the reference tiles are already well exposed, so any tone change counts as 'error' to PSNR even if it helps a hazy tile.

## Real tiles, no reference (classical output vs original)

| Tile | Sharpness before | Sharpness after | Colour shift (LAB a/b) |
|---|---|---|---|
| 1e44ba_google | 2028.6 | 9792.1 | 1.70 |
| 20d961_google | 2032.3 | 7923.5 | 0.90 |
| 3d9b33_google | 3630.2 | 13093.6 | 1.84 |
| 5aca8b_google | 6143.2 | 29275.5 | 1.57 |
| 666e65_google | 3896.4 | 17039.5 | 1.55 |
| 686199_google | 3969.7 | 21509.4 | 1.37 |
| 6c8fd5_google | 829.2 | 3507.4 | 1.37 |
| 7345ba_google | 7816.7 | 37562.7 | 1.41 |
| 761669_google | 3883.3 | 16706.7 | 1.46 |
| 7ecfdc_google | 2226.4 | 8442.7 | 1.81 |
| 8899da_google | 596.0 | 2872.9 | 2.39 |
| 8d05f2_google | 427.5 | 1366.8 | 1.05 |
| 90435e_google | 16478.8 | 72197.5 | 2.92 |
| 98fa77_google | 2990.4 | 10291.9 | 1.44 |
| a3a517_google | 9349.8 | 42421.3 | 1.40 |
| a90908_google | 8290.1 | 37537.7 | 1.46 |
| aba1eb_google | 928.0 | 4812.7 | 1.48 |
| b31756_google | 1599.1 | 7314.2 | 1.69 |
| c85563_google | 1345.6 | 4419.9 | 1.64 |
| cac04a_google | 1165.2 | 5093.3 | 1.69 |
| d87dbc_google | 7119.9 | 24944.0 | 2.16 |
| e42da1_google | 4670.3 | 22218.2 | 1.71 |
| f28a5f_google | 2980.3 | 9772.6 | 1.63 |
| f73de6_google | 441.5 | 2900.3 | 1.42 |
| **Mean** | **3951.6** | **17209.0** | **1.63** |

Higher sharpness is not automatically better: noise and halos also raise it.
