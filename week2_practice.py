import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Empêcher Matplotlib de chercher un écran graphique
import matplotlib
matplotlib.use('Agg')

# 1. Charger l'image du drone
image_name = 'real_drone_map.jpg'
field = cv2.imread(image_name)

if field is None:
    print(f"❌ Erreur : Impossible de charger l'image '{image_name}'.")
    exit()

# 2. Isoler la couleur verte (Masque)
hsv_field = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
lower_green = np.array([30, 35, 35])
upper_green = np.array([90, 255, 255])
mask = cv2.inRange(hsv_field, lower_green, upper_green)

# 3. LE SECRET EST ICI : Fusionner les branches avec un grand "Kernel"
# On passe de (5,5) à (25,25) pour boucher tous les trous du feuillage d'un coup
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

# On ajoute un petit nettoyage pour effacer les minuscules mauvaises herbes au sol
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open)

# 4. Trouver le centre (Distance Transform)
# Maintenant que le masque est un bloc solide, le centre sera parfait
dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
dist_smoothed = cv2.GaussianBlur(dist_transform, (15, 15), 0)

dist_visualization = cv2.normalize(dist_smoothed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heatmap_save = cv2.applyColorMap(dist_visualization, cv2.COLORMAP_HOT)

# 5. Comptage
kernel_max = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
local_max = cv2.dilate(dist_smoothed, kernel_max)

# On baisse légèrement le seuil à 0.1 pour être sûr de ne rater aucun petit arbre
peaks = (dist_smoothed == local_max) & (cleaned_mask > 0) & (dist_smoothed > 0.1 * dist_smoothed.max())
peaks_8u = np.uint8(peaks * 255)

contours, _ = cv2.findContours(peaks_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
tree_count = len(contours)

print(f"==================================================")
print(f"✅ SUCCÈS : {tree_count} arbres individuels détectés !")
print(f"==================================================")

# 6. Dessiner les cercles
field_detected = field.copy()
for c in contours:
    ((x, y), _) = cv2.minEnclosingCircle(c)
    cv2.circle(field_detected, (int(x), int(y)), 8, (0, 255, 0), -1)
    cv2.circle(field_detected, (int(x), int(y)), 30, (0, 255, 0), 2)

# 7. Sauvegarder
os.makedirs("output_results", exist_ok=True)
cv2.imwrite("output_results/cleaned_mask.jpg", cleaned_mask)
cv2.imwrite("output_results/tree_centers.jpg", heatmap_save)
cv2.imwrite("output_results/detected_trees.jpg", field_detected)

# 8. Dashboard
field_rgb = cv2.cvtColor(field_detected, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(field_rgb)
axes[0].set_title(f"1. Arbres détectés ({tree_count})")
axes[0].axis('off')

axes[1].imshow(cleaned_mask, cmap='gray')
axes[1].set_title("2. Masque solide (Branches fusionnées)")
axes[1].axis('off')

axes[2].imshow(dist_visualization, cmap='hot')
axes[2].set_title("3. Centres parfaits")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("output_results/global_dashboard.jpg")
print("💾 Rapport sauvegardé avec succès.")