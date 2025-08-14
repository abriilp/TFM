import pandas as pd
import numpy as np
import os
import cv2

def compute_psnr(img1, img2, data_range=255.0):
    """Compute PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        data_range (float): The range of the data, default is 255.0 for 8-bit images.

    Returns:
        float: PSNR value.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if images are identical
    return 20 * np.log10(data_range / np.sqrt(mse))

def compute_ssim(img1, img2, data_range=255.0):
    """Compute SSIM (Structural Similarity Index) between two images.

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        data_range (float): The range of the data, default is 255.0 for 8-bit images.

    Returns:
        float: SSIM value.
    """
    from skimage.metrics import structural_similarity as ssim
    # Change order of channels from (H, W, C) to (C, H, W) for SSIM computation if needed
    if img1.ndim == 3 and img1.shape[2] <= 4:
        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))
    return ssim(img1, img2, data_range=data_range, channel_axis=0)


dir_pred = "/home/mpilligua/Restormer/BestModelOut_val"
dir_gt = "/home/mpilligua/Restormer/BestModelOut_GT_val"

dir_ds = "/home/mpilligua/Restormer/metrics_val.csv"

df = pd.DataFrame(columns=['Image', 'PSNR', 'SSIM', 'Scene', 'Pan_in', 'Tilt_in', 'Pan_out', 'Tilt_out', 'idx_img'])

if os.path.exists(dir_ds):
    df = pd.read_csv(dir_ds)
    print(f"Loaded existing dataset with {len(df)} entries.")

for image in os.listdir(dir_pred):
    # try:
    pred_path = os.path.join(dir_pred, image)
    gt_path = os.path.join(dir_gt, image)

    img_pred = cv2.imread(pred_path)
    img_gt = cv2.imread(gt_path)
    
    if img_pred is None or img_gt is None:
        print(f"Error reading images for {image}. Skipping...")
        continue
    
    if image in df['Image'].values:
        print(f"Image {image} already processed. Skipping...")
        continue

    psnr_value = compute_psnr(img_pred, img_gt)
    ssim_value = compute_ssim(img_pred, img_gt)

    pan_in, tilt_in = int(image.split('_')[2]), int(image.split('_')[3])
    pan_out, tilt_out = int(image.split('_')[-2]), int(image.split('_')[-1].split('.')[0])

    scene = image.split('_')[1]
    idx_img = int(image.split('_')[0])

    print(f"Image: {image}, PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    # Save results to CSV with the idx_img as the index
    df = pd.concat([df, pd.DataFrame({
        'Image': [image],
        'PSNR': [psnr_value],
        'SSIM': [ssim_value],
        'Scene': [scene],
        'Pan_in': [pan_in],
        'Tilt_in': [tilt_in],
        'Pan_out': [pan_out],
        'Tilt_out': [tilt_out],
        'idx_img': [idx_img]
    })], ignore_index=True)
    
    if len(df) % 100 == 0:
        print(f"Processed {len(df)} images, saving to CSV...")
        df.to_csv(dir_ds, mode='a', header=not os.path.exists(dir_ds), index=False)
        
    # except Exception as e:
    #     print(f"Error processing image {image}: {e}")
        
df.to_csv(dir_ds, mode='a', header=not os.path.exists(dir_ds), index=False)
        