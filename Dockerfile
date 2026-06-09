# 1. Utiliser une version légère de Python
FROM python:3.10-slim

# 2. Définir le dossier de travail à l'intérieur du conteneur
WORKDIR /app

# 3. Copier tous les fichiers du projet (le script, l'image du drone, etc.) dans le conteneur
COPY . /app

# 4. Installer les bibliothèques requises (headless pour éviter les bugs d'affichage sous Docker)
RUN pip install --no-cache-dir numpy opencv-python-headless matplotlib

# 5. LA LIGNE MANQUANTE : Indiquer à Docker d'exécuter automatiquement votre script au démarrage
CMD ["python", "week2_practice.py"]