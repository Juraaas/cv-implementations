import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

boxes = np.array([
    [100, 50, 300, 200, 0.95],
    [110, 55, 310, 205, 0.88],
    [90, 45, 290, 195, 0.76],
    [400, 50, 600, 200, 0.91],
    [105, 52, 305, 202, 0.60],
])

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xlim(0, 700)
ax.set_ylim(250, 0)
ax.set_facecolor('#1a1a2e')
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for i, (x1, y1, x2, y2, score) in enumerate(boxes):
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, 
        linewidth=2, edgecolor=colors[i], facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-5, f'Box {chr(65+i)}: {score:.2f}',
            color=colors[i], fontsize=9, fontweight='bold')

ax.set_title('Before NMS', color='white', pad=10)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
plt.tight_layout()
#plt.savefig('outputs/before_nms.png', dpi=150, facecolor='#1a1a2e')
#plt.show()

def compute_iou(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2, score]
    returns: [0-1] float
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

box_A = boxes[0]
box_B = boxes[1]
box_D = boxes[3]

#print(f"IoU(A, B) = {compute_iou(box_A, box_B):.3f}")
#print(f"IoU(A, D) = {compute_iou(box_A, box_D):.3f}")
#print(f"IoU(A, A) = {compute_iou(box_A, box_A):.3f}")

def nms(boxes, iou_threshsold=0.5):
    """
    boxes: np.array shape (N, 5) - [x1, y1, x2, y2, score]
    iou_threshold: boxy z IoU > threshold względem wybranego są usuwane
    Zwraca: lista wybranych boxów
    """
    if len(boxes) == 0:
        return []

    # Krok 1: sortowanie malejąco po score
    sorted_idx = np.argsort(boxes[:, 4][::-1])
    boxes = boxes[sorted_idx]

    result = []

    while len(boxes) > 0:
        # Krok 2: bierzemy box z najwyższym score
        best = boxes[0]
        result.append(best)

        if len(boxes) == 1:
            break

        # Krok 3: IoU najlepszego z wszystkimi pozostałymi
        remaining = boxes[1:]
        iou_scores = np.array([compute_iou(best, b) for b in remaining])

        # Krok 4: zostaw tylko te które się NIE nakładają za bardzo
        boxes = remaining[iou_scores < iou_threshsold]

    return np.array(result)

print("Before NMS:", len(boxes), "boxes")
result = nms(boxes, iou_threshsold=0.5)
print("After NMS", len(result), "boxes")
for b in result:
    print(f"  [{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}] score={b[4]:.2f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (title, bxs) in zip(axes, [
    ('Before NMS', boxes),
    ('After NMS', result)
]):
    ax.set_xlim(0, 700)
    ax.set_ylim(250, 0)
    ax.set_facecolor('#1a1a2e')
    for i, (x1, y1, x2, y2, score) in enumerate(bxs):
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2.5, edgecolor=colors[i % len(colors)], facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{score:.2f}',
                color=colors[i % len(colors)], fontsize=10, fontweight='bold')
    ax.set_title(title, color='white', pad=8)
    ax.tick_params(color='white')
plt.tight_layout()
plt.savefig('outputs/nms_comparison.png', dpi=150, facecolor='#1a1a2e')
plt.show()

import torch
import torchvision

boxes_tensor = torch.tensor(boxes[:, :4], dtype=torch.float32)
scores_tensor = torch.tensor(boxes[:, 4], dtype=torch.float32)
keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
print("\nTorchvision NMS chose:", keep.numpy())
print("My NMS:", len(result), "boxes")


