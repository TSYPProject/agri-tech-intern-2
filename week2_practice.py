import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load your real drone picture
image_name = 'real_drone_map.jpg'
field = cv2.imread(image_name)

# 2. Find the green trees (Make a mask)
hsv_field = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
lower_green = np.array([30, 35, 35])
upper_green = np.array([90, 255, 255])
mask = cv2.inRange(hsv_field, lower_green, upper_green)

# 3. Fill the tiny black holes inside the white trees
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 4. Find the glowing center of each tree
dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
dist_visualization = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heatmap_save = cv2.applyColorMap(dist_visualization, cv2.COLORMAP_HOT)

# 5. Automatically create a folder and save the pictures inside it
os.makedirs("output_results", exist_ok=True)
cv2.imwrite("output_results/cleaned_mask.jpg", cleaned_mask)
cv2.imwrite("output_results/tree_centers.jpg", heatmap_save)
print("✅ Success! Pictures saved in the output_results folder.")

# 6. Show the pictures on your screen
field_rgb = cv2.cvtColor(field, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(field_rgb)
axes[0].set_title("1. Real Drone View")
axes[0].axis('off')

axes[1].imshow(cleaned_mask, cmap='gray')
axes[1].set_title("2. Cleaned Mask (No Holes)")
axes[1].axis('off')

axes[2].imshow(dist_visualization, cmap='hot')
axes[2].set_title("3. Glowing Centers")
axes[2].axis('off')

plt.tight_layout()
plt.show()