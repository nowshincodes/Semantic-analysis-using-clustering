import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import umap.umap_ as umap
from embedding import norm_embeddings, train_texts

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoencoder model
class SentenceAutoencoder(nn.Module):
    def __init__(self, input_dim=384, latent_dim=64, dropout_rate=0.3):
        super(SentenceAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

# Prepare data
inputs = torch.tensor(norm_embeddings, dtype=torch.float32).to(device)
dataset = TensorDataset(inputs)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, optimizer, loss
model = SentenceAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Pretrain autoencoder
for epoch in range(100):
    model.train()
    total_loss = 0
    total_mse_target = 0
    total_mse_others = 0

    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, _ = model(x)
        
        # Compute MSE with target (reconstruction loss)
        mse_target = criterion(x_recon, x)
        
        # Compute MSE with other samples
        mse_others = 0
        for i in range(x.size(0)):  # Use actual batch size
            current_recon = x_recon[i:i+1]
            other_inputs = torch.cat([x[:i], x[i+1:]], dim=0)
            mse_others += criterion(current_recon.expand(other_inputs.size(0), -1), other_inputs)
        mse_others = mse_others / (x.size(0) * (x.size(0) - 1)) if x.size(0) > 1 else 0
        
        # Total loss (using mse_target only for optimization)
        eps = 1e-8
      
        # loss = mse_target
        # loss = mse_target + (1/(mse_others + eps))

        alpha = 0.8
        beta = 0.1 

        loss = alpha * mse_target + beta * (1 / (mse_others + eps))

        loss.backward()
        optimizer.step()
        
        # Track losses
        total_mse_target += mse_target.item() * x.size(0)
        total_mse_others += mse_others * x.size(0) if x.size(0) > 1 else 0
        total_loss += loss.item() * x.size(0)
    
    # Print average losses per epoch
    avg_mse_target = total_mse_target / len(dataset)
    avg_mse_others = total_mse_others / len(dataset)
    avg_loss = total_loss / len(dataset)
    if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}, MSE Target: {avg_mse_target:.6f}, MSE Others: {avg_mse_others:.6f}")

# Extract latent vectors for all data
model.eval()
all_latents = []
with torch.no_grad():
    for batch in dataloader:
        x = batch[0].to(device)
        _, z = model(x)
        all_latents.append(z.cpu())
all_latents = torch.cat(all_latents).numpy()

print("Applying K-means clustering...")
n_clusters = 4 # Adjust based on your data or use elbow method
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(all_latents)


# Calculate Silhouette Score
sil_score = silhouette_score(all_latents, labels)
print(f"Silhouette Score (MyCluster): {sil_score:.4f}")


centers = kmeans.cluster_centers_
representatives = {}

for cluster_id in range(kmeans.n_clusters):
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_latents = all_latents[cluster_indices]
    
    # Compute distances to the cluster center
    dists = np.linalg.norm(cluster_latents - centers[cluster_id], axis=1)
    
    # Get indices of the 10 closest points
    closest_idxs = cluster_indices[np.argsort(dists)[:10]]
    
    # Collect the corresponding sentences
    representatives[cluster_id] = [train_texts[idx] for idx in closest_idxs]

# Print representative sentences per cluster
for cluster_id, sents in representatives.items():
    print(f"\nCluster {cluster_id} - Top 10 representative sentences:")
    for i, sent in enumerate(sents, 1):
        print(f"{i:2d}. {sent}")


# Visualization with UMAP
print("Visualizing clusters with UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
z_2d = reducer.fit_transform(all_latents)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=10)
plt.colorbar(scatter)
plt.title('UMAP Visualization of MyCluster Clusters')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()

from torchsummary import summary
model = SentenceAutoencoder()
summary(model, input_size=(1, 384))
