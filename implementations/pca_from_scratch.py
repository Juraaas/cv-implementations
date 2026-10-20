import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200

data = np.random.randn(n, 2)
data[:, 0] *= 3
data[:, 1] *= 0.8

angle = np.radians(60)
R = np.array([[np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])
data = data @ R.T


def pca_from_scratch(X, n_components=2):
    mean = X.mean(axis=0)
    X_centered = X - mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance = (S ** 2) / (len(X) - 1)
    explained_ratio = explained_variance / explained_variance.sum()

    components = Vt[:n_components]
    X_projected = X_centered @ components.T

    return X_projected, components, explained_ratio, mean

X_proj, components, exp_ratio, mean = pca_from_scratch(data, n_components=2)

print(f"PC1 explains: {exp_ratio[0]*100:.1f}% of variance")
print(f"PC2 explains: {exp_ratio[1]*100:.1f}% of variance")
print(f"Direction of PC1: {components[0]}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax = axes[0]
ax.scatter(data[:, 0], data[:, 1], alpha=0.4, s=15, color='steelblue')
scale = 2.5
for i, (comp, ratio) in enumerate(zip(components, exp_ratio)):
    ax.annotate('', xy=mean + scale * ratio * comp, xytext=mean, 
                arrowprops=dict(arrowstyle='->', color=['red', 'orange'][i], lw=2.5))
    ax.text(*(mean + scale * ratio * comp * 1.1),
            f'PC{i+1} ({ratio*100:.0f}%)', 
            color=['red', 'orange'][i], fontsize=10, fontweight='bold')
ax.set_aspect('equal')
ax.set_title('Original data + principal components')
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.4, s=15, color='coral')
ax.set_xlabel(f'PC1 ({exp_ratio[0]*100:.0f}%)')
ax.set_ylabel(f'PC2 ({exp_ratio[1]*100:.0f}%)')
ax.set_title('Data in PCA space')
ax.grid(alpha=0.3)

X_1d = data - mean
X_reconstructed = (X_1d @ components[0:1].T) @ components[0:1] + mean

ax = axes[2]
ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=15, color='steelblue', label='Original')
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.6, s=15, color='red', label='Reconstructed (1D)')
ax.set_aspect('equal')
ax.set_title('Reconstruction from principal component')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/pca_from_scratch.png', dpi=150)
plt.show()

from sklearn.decomposition import PCA
pca_sklearn = PCA(n_components=2)
pca_sklearn.fit(data)
print("\nSKLearn PC1:", pca_sklearn.components_[0])
print("My PC1:", components[0])
print("Explained ratio sklearn:", pca_sklearn.explained_variance_ratio_)
print("Explained ratio mine:", exp_ratio)