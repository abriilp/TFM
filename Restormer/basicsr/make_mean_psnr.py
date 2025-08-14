import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
csv_path = "/home/mpilligua/Restormer/metrics_val.csv"  # Replace with your actual file path
df = pd.read_csv(csv_path)

# Ignore infinite values for mean calculation
mean_psnr = df.loc[np.isfinite(df["PSNR"]), "PSNR"].mean()
mean_ssim = df.loc[np.isfinite(df["SSIM"]), "SSIM"].mean()
print("Mean PSNR is", mean_psnr, f"\nMean SSIM is {mean_ssim}")


df['PSNR'] = df['PSNR'].replace([np.inf, 'inf', float('inf')], 100.0)

# Ensure correct column names
df.columns = ['Image', 'PSNR', 'SSIM', 'Scene', 'Pan_in', 'Tilt_in', 'Pan_out', 'Tilt_out', 'idx_img']

# Clean and preprocess
# df = df.replace("inf", np.nan)
df["PSNR"] = pd.to_numeric(df["PSNR"])
df["SSIM"] = pd.to_numeric(df["SSIM"])
# df = df.dropna(subset=["PSNR", "SSIM"])

# Combine pan/tilt into tuple keys
df["in_key"] = list(zip(df["Pan_in"], df["Tilt_in"]))
df["out_key"] = list(zip(df["Pan_out"], df["Tilt_out"]))

# Get only used directions
unique_in = sorted(df["in_key"].unique())
unique_out = sorted(df["out_key"].unique())
print(f"Unique input views: {unique_in}"
      f"\nUnique output views: {unique_out}")

# Map to indices
in_to_idx = {k: i for i, k in enumerate(unique_in)}
out_to_idx = {k: i for i, k in enumerate(unique_out)}

print(f"Input view to index mapping: {in_to_idx}"
      f"\nOutput view to index mapping: {out_to_idx}")

# Init matrices
mean_psnr = np.full((len(unique_in), len(unique_out)), np.nan)
mean_ssim = np.full((len(unique_in), len(unique_out)), np.nan)

# Fill matrices with means
for (in_key, out_key), group in df.groupby(["in_key", "out_key"]):
    print(f"Processing in_key: {in_key}, out_key: {out_key}")
    i = in_to_idx[in_key]
    j = out_to_idx[out_key]
    mean_psnr[i, j] = group["PSNR"].mean()
    mean_ssim[i, j] = group["SSIM"].mean()

# Plot heatmaps
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(mean_psnr, xticklabels=[str(p) for p in unique_out], yticklabels=[str(p) for p in unique_in],
            annot=True, fmt=".2f", cmap="viridis", ax=axes[0], vmin=20, vmax=40)
axes[0].set_title("Mean PSNR from [Pan_in, Tilt_in] → [Pan_out, Tilt_out]")
axes[0].set_xlabel("Output View")
axes[0].set_ylabel("Input View")

sns.heatmap(mean_ssim, xticklabels=[str(p) for p in unique_out], yticklabels=[str(p) for p in unique_in],
            annot=True, fmt=".3f", cmap="magma", ax=axes[1], vmin=0, vmax=1)
axes[1].set_title("Mean SSIM from [Pan_in, Tilt_in] → [Pan_out, Tilt_out]")
axes[1].set_xlabel("Output View")
axes[1].set_ylabel("Input View")

plt.tight_layout()
plt.savefig("/home/mpilligua/Restormer/mean_psnr_ssim_heatmaps_val.png", dpi=300)