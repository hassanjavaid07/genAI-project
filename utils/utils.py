import os
import tqdm
import random
import numpy as np
import wget, tarfile
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def sanitizeFilename(filepath):
    # To prevent Windows OS error
    directory, filename = os.path.split(filepath)
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return directory, filename


def downloadArtBenchData(download_directory, 
                         extract_directory, 
                         download=False, 
                         explore=True,
                         displaySingle=True):
    
    print("Downloading artbench-10-imagefolder-split.tar ...")
    wget.download(url="https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
                out = download_directory)

    
def extractData(download_directory,
                extract_directory,
                f_name):
    
    tar_file_path = os.path.join(download_directory, f'{f_name}.tar')
    print("\nUncompressing the tar file...")
    with tarfile.open(tar_file_path) as tar:
        members = tar.getmembers()
        for member in tqdm.tqdm(members, desc="Extracting", unit="file"):
            try:
                directory, filename = sanitizeFilename(member.name)
                member.name = os.path.join(directory, filename)
                tar.extract(member, path=extract_directory)
            except OSError as e:
                print(f"Skipping file {member.name} due to OSError: {e}") # Catch the error for invalid filenames
    
    print("Tar file uncompressed to:", extract_directory)



def exploringData(extract_directory):
    for dirpath, dirnames, filenames in os.walk(Path(extract_directory)):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")



def displaySingleSample(image_path, class_name):
    image_path = Path(image_path)
    image_path_list = list(image_path.glob(f"train/{class_name}/*.jpg"))
    random_image_path = random.choice(image_path_list)
    image_name = random_image_path.name
    image_class = random_image_path.parent.stem
    img = Image.open(random_image_path)
    print(f"Image name: {image_name}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")
    img.show()    
    return img


def imgDisp(img):
    img = img / 2 + 0.5  # Unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # Convert from (C, H, W) to (H, W, C)
    plt.axis('off')

def displayImageSamples(data_loader, classes, num_images=6):
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    cols = 3  
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(12, 4 * rows))  
    for i in range(num_images): 
        plt.subplot(rows, cols, i + 1)
        imgDisp(images[i])
        plt.title(classes[labels[i]])
    plt.tight_layout()  
    plt.show()


def makeTSNEPlot(image_features, labels, 
                 n_components=2, 
                 random_state=42, 
                 save_path=None):
    image_features_np = image_features.numpy()
    labels_np = labels.numpy()

    tsne = TSNE(n_components=n_components, random_state=random_state)
    features_2d = tsne.fit_transform(image_features_np)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_np, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
