import os
import torch
import lpips
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# Paths
original_root = "/workspace/datasets/KODAK"          # Original images: ./original_CLIC/<image_name>.png
reconstruction_root = "./recon_results/KODAK/"    # Reconstructed images: ./recon_results/<image_name>/final_recon/qXXXX.png
dataset_name = "KODAK"

# original_root = "/workspace/datasets/clic_prof_valid"          # Original images: ./original_CLIC/<image_name>.png
# reconstruction_root = "./recon_results/clic_prof_valid/"    # Reconstructed images: ./recon_results/<image_name>/final_recon/qXXXX.png
# dataset_name = "clic_prof_valid"

# original_root = "/workspace/datasets/TECNICK100"
# reconstruction_root = "./recon_results/TECNICK100/"
# dataset_name = "TECNICK100"


# LPIPS model (AlexNet backbone)
lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

# Image transform
to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

results = []

# Iterate through reconstructed image folders
for folder in os.listdir(reconstruction_root):
    print(f"Processing {folder}...")
    recon_folder_path = os.path.join(reconstruction_root, folder)  # folder includes extension, e.g., abc.png

    # Original image path: match by folder name (folder includes extension)
    original_img_path = os.path.join(original_root, folder)
    if not os.path.isfile(original_img_path):
        # print(original_img_path)
        print(f"Original image not found for {folder}, skipping...")
        continue
    if not os.path.isdir(recon_folder_path):
        print(f"Reconstruction folder missing for {folder}, skipping...")
        continue

    # Load original image
    orig_img = Image.open(original_img_path).convert("RGB")
    orig_tensor = to_tensor(orig_img).unsqueeze(0).cuda()

    # Iterate through all reconstructed images (qXX_pYY.png)
    for recon_file in sorted(os.listdir(recon_folder_path)):
        if not recon_file.endswith(".png"):
            continue

        # Extract full rate_point (qXX_pYY)
        rate_point = os.path.splitext(recon_file)[0]
        recon_img_path = os.path.join(recon_folder_path, recon_file)

        # Load reconstructed image
        recon_img = Image.open(recon_img_path).convert("RGB")
        recon_tensor = to_tensor(recon_img).unsqueeze(0).cuda()

        # Compute LPIPS
        lpips_val = lpips_fn(orig_tensor, recon_tensor).item()

        results.append({
            "image": folder,
            "rate_point": rate_point,
            "lpips": lpips_val
        })

# Convert to DataFrame
df = pd.DataFrame(results)

# Compute average LPIPS per rate_point
avg_df = df.groupby("rate_point")[["lpips"]].mean().reset_index()

# Save to CSV
os.makedirs(f"./metrics/{dataset_name}", exist_ok=True)
csv_path = f"./metrics/{dataset_name}/lpips_rate_results.csv"
avg_df.to_csv(csv_path, index=False)
print(f"Saved average LPIPS results to {csv_path}")
