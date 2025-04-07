from common import *
from ocr_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_word_images_dataset(output_dir):
    nltk.download('words')
    word_list = words.words()
    num_words = 100000
    sampled_words = random.sample(word_list, num_words)
    img_width, img_height = 256, 64
    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.load_default()

    for i, word in enumerate(sampled_words):
        image = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(image)

        bbox = draw.textbbox((0, 0), word, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = (img_width - text_width) / 2
        text_y = (img_height - text_height) / 2
        draw.text((text_x, text_y), word, fill='black', font=font)
        image.save(os.path.join(output_dir, f"{word}_{i}.png"))

        print("Dataset created successfully!")

class WordImagesDataset(Dataset):
    def __init__(self, folder_path):
        self.folder = folder_path
        self.image_files = [path for path in os.listdir(folder_path) if path.endswith(".png")]
        self.transform = transforms.ToTensor()  
        self.encoder = character_encoder()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.folder, img_filename)
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        image = np.where(image > 127, 0, 1)
        if self.transform:
            image = self.transform(image)
        image = image.float()
        label = img_filename.split('_')[0]  
        label_indices = torch.tensor([self.encoder[char] for char in label], dtype=torch.long)
        return image, label_indices

def batch_preprocess(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)

    labels = [label[:32].clone().detach() if len(label) > 32 else label.clone().detach() for label in labels]    
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=52)
    if padded_labels.size(1) < 32:
        padded_labels = torch.nn.functional.pad(padded_labels, (0, 32 - padded_labels.size(1)), value=52)
    else:
        padded_labels = padded_labels[:, :32]

    return images, padded_labels

def random_baseline(labels):
    correct_chars = 0
    total_chars = 0
    for true_seq in labels:
        random_seq = ''.join(random.choices(string.ascii_lowercase, k=len(true_seq)))
        correct_chars += sum([1 for rand_char, true_char in zip(random_seq, true_seq) if rand_char == true_char])
        total_chars += len(true_seq)

    avg_correct_chars = correct_chars / total_chars
    return avg_correct_chars

## Dataset Preparation
folder_path = "../../data/external/word_images"
# generate_word_images_dataset(output_dir=folder_path)
dataset = WordImagesDataset(folder_path=folder_path)

train_ratio = 0.8
test_ratio = 0.2

train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=batch_preprocess)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_preprocess)

## Model Training
model = OCRModel(num_classes=53)  
model.train_model(train_loader, val_loader, epochs=10, lr=0.001, device=device)

## Comparision with Random Baseline
labels = [path.split('_')[0] for path in os.listdir(folder_path) if path.endswith(".png")]
print(f"ANCC Based On Random Baseline: {random_baseline(labels)}")
print(f"ANCC Based On OCR Model Predictions : {model.evaluate(val_loader, device)}")
