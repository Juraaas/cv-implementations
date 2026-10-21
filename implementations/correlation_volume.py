import numpy as np

def compute_correlation_volume(feat1, feat2):
    """
    feat1: (H, W, C) - image 1 feature map
    feat2: (H, W, C) - image 2 feature map
    Returns: (H, W, H, W) — for every pixel pair, returns their similarity
    """
    H, W, C = feat1.shape
    f1 = feat1.reshape(H*W, C)
    f2 = feat2.reshape(H*W, C)

    # Similarity matrix - dot product every pixel
    corr = f1 @ f2.T

    return corr.reshape(H, W, H, W)

np.random.seed(42)
H, W, C = 4, 4, 8

feat1 = np.random.randn(H, W, C)
feat2 = np.random.randn(H, W, C)

feat2[1, 1] = feat1[1, 1]

corr = compute_correlation_volume(feat1, feat2)
print(f"Correlation volume shape: {corr.shape}")

corr_for_pixel_11 = corr[1, 1]
best_match = np.unravel_index(corr_for_pixel_11.argmax(), (H, W))
print(f"Pixel (1,1) best matches: {best_match}")