"""
Usage:
    python compare_screenshots.py --reference baseline_fp32 --compare position_fp16
    python compare_screenshots.py -r baseline_fp32 -c position_fp16 --output results.csv
    python compare_screenshots.py --reference baseline_fp32 -compare position_fp16 --diff --diff-amplify
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except ImportError:
    print("Error: scikit-image is required. Install with: pip install scikit-image")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


def load_ppm(filepath: str) -> np.ndarray:
    img = Image.open(filepath)
    return np.array(img)


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # SSIM works on grayscale or channel-by-channel
    # For RGB, we compute per-channel and average
    if len(img1.shape) == 3:
        return ssim(img1, img2, channel_axis=2, data_range=255)
    return ssim(img1, img2, data_range=255)


def extract_frame_number(filename: str) -> int:
    match = re.search(r'frame(\d+)\.ppm$', filename)
    if match:
        return int(match.group(1))
    return -1


def find_matching_frames(screenshots_dir: str, ref_prefix: str, cmp_prefix: str) -> list:
    ref_files = {}
    cmp_files = {}

    for f in os.listdir(screenshots_dir):
        if f.startswith(ref_prefix) and f.endswith('.ppm'):
            frame_num = extract_frame_number(f)
            if frame_num >= 0:
                ref_files[frame_num] = f
        elif f.startswith(cmp_prefix) and f.endswith('.ppm'):
            frame_num = extract_frame_number(f)
            if frame_num >= 0:
                cmp_files[frame_num] = f

    common_frames = sorted(set(ref_files.keys()) & set(cmp_files.keys()))

    pairs = []
    for frame_num in common_frames:
        pairs.append((frame_num, ref_files[frame_num], cmp_files[frame_num]))

    return pairs


def generate_diff_image(img1: np.ndarray, img2: np.ndarray, amplify: float = 10.0) -> np.ndarray:
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    diff = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    return diff


def main():
    parser = argparse.ArgumentParser(
        description='Compare screenshots using PSNR and SSIM metrics'
    )
    parser.add_argument(
        '-r', '--reference',
        required=True,
        help='Reference image prefix (e.g., baseline_fp32)'
    )
    parser.add_argument(
        '-c', '--compare',
        required=True,
        help='Comparison image prefix (e.g., position_fp16)'
    )
    parser.add_argument(
        '-d', '--directory',
        default='screenshots',
        help='Screenshots directory (default: screenshots)'
    )
    parser.add_argument(
        '-o', '--output',
        default='comparison_results.csv',
        help='Output CSV filename (default: comparison_results.csv)'
    )
    parser.add_argument(
        '--diff',
        action='store_true',
        help='Generate difference images'
    )
    parser.add_argument(
        '--diff-amplify',
        type=float,
        default=10.0,
        help='Amplification factor for difference images (default: 10.0)'
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    screenshots_dir = script_dir / args.directory

    if not screenshots_dir.exists():
        print(f"Error: Directory not found: {screenshots_dir}")
        sys.exit(1)

    pairs = find_matching_frames(str(screenshots_dir), args.reference, args.compare)

    if not pairs:
        print(f"Error: No matching frames found for prefixes '{args.reference}' and '{args.compare}'")
        sys.exit(1)

    print(f"Found {len(pairs)} matching frame pairs")
    print(f"Reference: {args.reference}")
    print(f"Compare:   {args.compare}")
    print("-" * 70)

    if args.diff:
        diff_dir = screenshots_dir / 'diff'
        diff_dir.mkdir(exist_ok=True)

    results = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0

    for frame_num, ref_file, cmp_file in pairs:
        ref_path = screenshots_dir / ref_file
        cmp_path = screenshots_dir / cmp_file

        ref_img = load_ppm(str(ref_path))
        cmp_img = load_ppm(str(cmp_path))

        if ref_img.shape != cmp_img.shape:
            print(f"Warning: Frame {frame_num} dimension mismatch: {ref_img.shape} vs {cmp_img.shape}")
            continue

        mse_val = calculate_mse(ref_img, cmp_img)
        psnr_val = calculate_psnr(ref_img, cmp_img)
        ssim_val = calculate_ssim(ref_img, cmp_img)

        results.append({
            'frame': frame_num,
            'mse': mse_val,
            'psnr': psnr_val,
            'ssim': ssim_val
        })

        total_mse += mse_val
        total_psnr += psnr_val if psnr_val != float('inf') else 100.0
        total_ssim += ssim_val

        psnr_str = f"{psnr_val:.2f}" if psnr_val != float('inf') else "inf"
        print(f"Frame {frame_num:5d}: MSE={mse_val:10.4f}  PSNR={psnr_str:>8} dB  SSIM={ssim_val:.6f}")

        if args.diff:
            diff_img = generate_diff_image(ref_img, cmp_img, args.diff_amplify)
            diff_path = diff_dir / f"diff_frame{frame_num}.png"
            Image.fromarray(diff_img).save(str(diff_path))

    n = len(results)
    if n > 0:
        avg_mse = total_mse / n
        avg_psnr = total_psnr / n
        avg_ssim = total_ssim / n

        print("-" * 70)
        print(f"AVERAGE:       MSE={avg_mse:10.4f}  PSNR={avg_psnr:8.2f} dB  SSIM={avg_ssim:.6f}")
        print()

        print("Quality Assessment:")
        if avg_psnr >= 40:
            print(f"  PSNR {avg_psnr:.1f} dB: Excellent - virtually indistinguishable")
        elif avg_psnr >= 30:
            print(f"  PSNR {avg_psnr:.1f} dB: Good - minor differences, acceptable quality")
        elif avg_psnr >= 20:
            print(f"  PSNR {avg_psnr:.1f} dB: Fair - noticeable differences")
        else:
            print(f"  PSNR {avg_psnr:.1f} dB: Poor - significant visual degradation")

        if avg_ssim >= 0.99:
            print(f"  SSIM {avg_ssim:.4f}: Excellent structural similarity")
        elif avg_ssim >= 0.95:
            print(f"  SSIM {avg_ssim:.4f}: Good structural similarity")
        elif avg_ssim >= 0.90:
            print(f"  SSIM {avg_ssim:.4f}: Acceptable structural similarity")
        else:
            print(f"  SSIM {avg_ssim:.4f}: Poor structural similarity")

    output_path = script_dir / args.output
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'mse', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(results)
        if n > 0:
            writer.writerow({
                'frame': 'AVERAGE',
                'mse': avg_mse,
                'psnr': avg_psnr,
                'ssim': avg_ssim
            })

    print(f"\nResults saved to: {output_path}")

    if args.diff:
        print(f"Difference images saved to: {diff_dir}")


if __name__ == '__main__':
    main()
