import clip
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils.utils import * 

# Change paths according to your system
DATA_ROOT = Path(r'data')
ROOT_DIR = Path(r'')

download_directory = os.path.join(DATA_ROOT, 'artbench-10-download')
extract_directory = os.path.join(DATA_ROOT, 'artbench-10')
features_directory = os.path.join(ROOT_DIR, 'clip-features')
outputs_directory = os.path.join(ROOT_DIR, 'output')
    

def getData(images_path):  
    download = False # Set to True to start downloading
    if download:
        downloadArtBenchData(download_directory, extract_directory)
    extract = False # Set to True to start extracting
    if extract:
        extractData(download_directory, extract_directory, f_name)
    # Optional Parameters   
    explore, displaySingle = True, True
    if explore:
        exploringData(extract_directory)
    if displaySingle:
        displaySingleSample(images_path, class_name='impressionism')


def loadCLIPModel(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess



def getClipFeatures(data_loader, 
                    model, 
                    device):
    all_image_features = []
    all_labels = []

    for images, labels in tqdm.tqdm(data_loader, desc="Processing batches", leave=True):
        images = images.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)

        all_image_features.append(image_features.cpu())
        all_labels.append(labels)

    all_image_features = torch.cat(all_image_features)  # [N, feature_dim]
    all_labels = torch.cat(all_labels)

    return all_image_features, all_labels

def train():
      pass


if __name__=="__main__":
    os.makedirs(download_directory, exist_ok=True)
    os.makedirs(extract_directory, exist_ok=True)
    os.makedirs(features_directory, exist_ok=True)
    os.makedirs(outputs_directory, exist_ok=True)
    
    
    f_name = 'artbench-10-imagefolder-split'
    images_path = os.path.join(extract_directory, f_name)
    getData(images_path=images_path)

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Load ArtBench Dataset
    train_path = os.path.join(images_path, 'train')   
    train_data = ImageFolder(root=train_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Print class info & mappings
    classInfo = False
    if classInfo:    
        print(f'Number of classes: {len(train_data.classes)}')
        print(f'Classes: {train_data.classes}')
        print(f'Number of images: {len(train_data)}')
        print(f"Class-to-Index Mapping: {train_data.class_to_idx}")  # Mapping class name to label

        for images, labels in train_loader:
            print(f"Image batch shape: {images.shape}")
            print(f"Label batch shape: {labels.shape}")
            print(f"Labels: {labels[:10]}")
            break
    
    displayBatchSamples = False
    if displayBatchSamples:
        displayImageSamples(train_loader, 
                            classes=train_data.classes, 
                            num_images=12)
    
    # Load CLIP model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = loadCLIPModel(device=device)

    # Display model properties
    print("CLIP Model Loaded:")
    print(f"Model Architecture: {clip_model.__class__.__name__}")
    print(f"Input Image Size: {preprocess.transforms[0].size}")
    print(f"Number of Parameters: {sum(p.numel() for p in clip_model.parameters())}")
    
    clipExtract = False # CHANGE to FALSE if feature tensor already extracted
    features_path = os.path.join(features_directory, 'artbench-features-with-labels.pt')
    if clipExtract: # Extract
        clip_features, data_labels = getClipFeatures(train_loader,
                                                    clip_model,
                                                    device=device)
        torch.save((clip_features, data_labels), features_path)
        print("Features saved with labels.")
    else: # Load if features already extracted
        print('Features already present. Loading...')
        image_features, labels = torch.load(features_path)
        print('Featues loaded.')
        print(f"Image Features Shape: {image_features.shape}")  # [N, feature_dim]
        print(f"Labels Shape: {labels.shape}")  # [N]

    visualize = False
    if visualize:
        save_path = os.path.join(outputs_directory, 'artbench-tnse-plot.png')
        makeTSNEPlot(image_features, labels, 
                     n_components=2, 
                     random_state=42,
                     save_path=save_path)