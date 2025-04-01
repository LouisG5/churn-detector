import os
import zipfile

def zip_images_in_batches(source_folder, batch_size=200, output_folder="batches"):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

    zip_paths = []
    for idx, batch in enumerate(batches):
        zip_filename = os.path.join(output_folder, f"batch_{idx + 1:03d}.zip")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for image in batch:
                zipf.write(os.path.join(source_folder, image), arcname=image)
        zip_paths.append(zip_filename)

    return zip_paths

# === Lance le script ===
if __name__ == "__main__":
    folder = "/Users/sebastianonise/Documents/ML/milestone_box/train/images"  # <-- à adapter
    output = "/Users/sebastianonise/Documents/ML/datasets/batches"
    zips = zip_images_in_batches(folder, batch_size=20, output_folder=output)
    print("Fichiers ZIP créés :", zips)
