from sentence_transformers import SentenceTransformer
from pre import train_dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import normalize

# Load and encode sentences
train_texts = train_dataset["sentence"]
true_labels = train_dataset["label"]

# Define file path for saving embeddings
embeddings_file = r"V:/CSE425project/embeddings.npy"

# Check if embeddings already exist
if os.path.exists(embeddings_file):
    print("Loading cached embeddings...")
    train_embeddings = np.load(embeddings_file)
else:
    print("Generating embeddings...")
    # Load and encode sentences
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_embeddings = model.encode(train_texts, batch_size=32, show_progress_bar=True)
    # Save embeddings to disk
    np.save(embeddings_file, train_embeddings)
    print(f"Embeddings saved to {embeddings_file}")



# L2 normalize embeddings
norm_embeddings = normalize(train_embeddings, norm='l2')
# Extract labels (adjust 'label' to the correct column name in your dataset)

labels = train_dataset["label"]  # Replace 'label' with your actual label column name

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
embeddings_2d = tsne.fit_transform(train_embeddings)

# Visualize
plt.figure(figsize=(10, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Label')
plt.title("t-SNE of Sentence Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()