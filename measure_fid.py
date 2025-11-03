import os
import csv
from PIL import Image
from torch_fidelity import calculate_metrics

# ---- Utilities ----
def ensure_rgb(folder):
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    print(f"Converting {fname} from {img.mode} to RGB")
                    img.convert("RGB").save(path)
        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
            print(f"[Error] Cannot process {fname}: {e}")

def resize_images_to_new_folder(src_folder, dst_folder, size=(299, 299)):
    os.makedirs(dst_folder, exist_ok=True)
    for fname in sorted(os.listdir(src_folder)):
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)

        try:
            if os.path.exists(dst_path):
                try:
                    with Image.open(dst_path) as dst_img:
                        if dst_img.size == size and dst_img.mode == "RGB":
                            continue
                except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
                    print(f"[Error] Cannot process {fname}: {e}")

            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize(size, Image.BILINEAR)
                img.save(dst_path)

        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
            print(f"[Error] Cannot resize {fname}: {e}")

def center_crop_fixed_size_images_to_new_folder(src_folder, dst_folder, crop_size=(299, 299)):
    """
    Center-crop a fixed-size rectangle (default 299x299) from the original image, no resizing.
    If the image is smaller than crop_size, it will be skipped with a warning.
    """
    os.makedirs(dst_folder, exist_ok=True)
    crop_w, crop_h = crop_size

    for fname in sorted(os.listdir(src_folder)):
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)

        try:
            if os.path.exists(dst_path):
                try:
                    with Image.open(dst_path) as dst_img:
                        if dst_img.size == crop_size and dst_img.mode == "RGB":
                            continue
                except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
                    print(f"[Warning] Could not verify {dst_path}, reprocessing: {e}")

            with Image.open(src_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                if w < crop_w or h < crop_h:
                    print(f"[Skip] {fname}: too small for {crop_w}x{crop_h} crop (size={w}x{h})")
                    continue

                left = (w - crop_w) // 2
                top = (h - crop_h) // 2
                img = img.crop((left, top, left + crop_w, top + crop_h))
                img.save(dst_path)


        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
            print(f"[Error] Cannot center-crop {fname}: {e}")



# ---- Config ----
# input1_original = "/workspace/datasets/coco/val2017"
# input2_root = "/workspace/2025_ToMM_quality_scalable/LIC_TCM-main_8levels/recon_results_for_fid/val2017"
# input2_root = "/workspace/2025_ToMM_quality_scalable/CTC-main/results/recon"

input1_original = "/workspace/datasets/DIV2K"
# input2_root = "/workspace/2025_ToMM_quality_scalable/LIC_TCM-main_8levels/recon_results_for_fid/DIV2K"
input2_root = "/workspace/2025_ToMM_quality_scalable/CTC-main/results_DIV2k/recon"

# ---- Prepare GT (both pipelines) ----
ensure_rgb(input1_original)

input1_resized = input1_original.rstrip("/") + "_resized"       # plain resize
input1_ccrop   = input1_original.rstrip("/") + "_ccrop299"      # center-crop + resize

resize_images_to_new_folder(input1_original, input1_resized)
center_crop_fixed_size_images_to_new_folder(input1_original, input1_ccrop)

# ---- Recon folders ----
input2_dirs = sorted([
    os.path.join(input2_root, d)
    for d in os.listdir(input2_root)
    if (
        os.path.isdir(os.path.join(input2_root, d)) and
        not (d.endswith("_ccrop299") or d.endswith("_resized"))
    )
])

# ---- CSV output ----
csv_path = os.path.join(input2_root, "fid_scores.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["recon_folder", "FID_resize", "FID_center_crop"])

    for input2 in input2_dirs:
        recon_name = os.path.basename(input2)
        print(f"\n===> Calculating FID for: {recon_name}")

        # Prepare recon (both pipelines)
        ensure_rgb(input2)
        input2_resized = input2.rstrip("/") + "_resized"
        input2_ccrop   = input2.rstrip("/") + "_ccrop299"
        resize_images_to_new_folder(input2, input2_resized)
        center_crop_fixed_size_images_to_new_folder(input2, input2_ccrop)

        try:
            # 1) FID with plain resize → resize vs resize
            metrics_resize = calculate_metrics(
                input1=input1_resized,
                input2=input2_resized,
                cuda=True,
                fid=True,
                isc=False,
                kid=False,
                batch_size=64,
                save_cpu_ram=True,
                input1_model_resize=False,
                input2_model_resize=False,
            )
            fid_resize = float(metrics_resize["frechet_inception_distance"])

            # 2) FID with center-crop pipeline → ccrop vs ccrop
            metrics_ccrop = calculate_metrics(
                input1=input1_ccrop,
                input2=input2_ccrop,
                cuda=True,
                fid=True,
                isc=False,
                kid=False,
                batch_size=64,
                save_cpu_ram=True,
                input1_model_resize=False,
                input2_model_resize=False,
            )
            fid_ccrop = float(metrics_ccrop["frechet_inception_distance"])

            print(f"{recon_name}: FID_resize={fid_resize:.4f} | FID_center_crop={fid_ccrop:.4f}")
            writer.writerow([recon_name, f"{fid_resize:.4f}", f"{fid_ccrop:.4f}"])


        except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
            print(f"[Error] Failed on {recon_name}: {e}")
            writer.writerow([recon_name, "ERROR", "ERROR"])
