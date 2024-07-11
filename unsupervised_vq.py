import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# 데이터 디렉토리 설정
data_dir = './RNN_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 그래프를 저장할 폴더 생성
graphs_dir = './CIFAR/GRU_graphs'
os.makedirs(graphs_dir, exist_ok=True)

# MNIST 데이터 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

train_data = next(iter(train_loader))
test_data = next(iter(test_loader))

x_train, y_train = train_data[0].numpy(), train_data[1].numpy()
x_test, y_test = test_data[0].numpy(), test_data[1].numpy()

# 0-9의 각 숫자 이미지에서 1000개씩 샘플링하여 사용
num_samples = 1000
x_train_samples = []
y_train_samples = []

for i in range(10):
    idx = np.where(y_train == i)[0][:num_samples]
    x_train_samples.append(x_train[idx])
    y_train_samples.append(y_train[idx])

x_train_samples = np.concatenate(x_train_samples)
y_train_samples = np.concatenate(y_train_samples)

# 0-99 이미지 생성
def create_combined_images(x_samples, num_samples_per_class_train=200, num_samples_per_class_test=20):
    combined_images_train = []
    combined_labels_train = []
    combined_images_test = []
    combined_labels_test = []
    for i in range(100):
        for _ in range(num_samples_per_class_train):
            num1 = i // 10
            num2 = i % 10
            img1 = x_samples[num1 * num_samples + np.random.randint(num_samples)]
            img2 = x_samples[num2 * num_samples + np.random.randint(num_samples)]
            combined_img = np.hstack((img1[0], img2[0]))
            combined_label = num1 * 10 + num2
            combined_images_train.append(combined_img)
            combined_labels_train.append(combined_label)
        for _ in range(num_samples_per_class_test):
            num1 = i // 10
            num2 = i % 10
            img1 = x_samples[num1 * num_samples + np.random.randint(num_samples)]
            img2 = x_samples[num2 * num_samples + np.random.randint(num_samples)]
            combined_img = np.hstack((img1[0], img2[0]))
            combined_label = num1 * 10 + num2
            combined_images_test.append(combined_img)
            combined_labels_test.append(combined_label)
    return (np.array(combined_images_train), np.array(combined_labels_train),
            np.array(combined_images_test), np.array(combined_labels_test))

# 20,000개의 training dataset 및 2,000개의 testing dataset 생성
combined_images_train, combined_labels_train, combined_images_test, combined_labels_test = create_combined_images(x_train_samples, num_samples_per_class_train=200, num_samples_per_class_test=20)

# 이미지 평탄화
combined_images_train_flat = combined_images_train.reshape(combined_images_train.shape[0], -1)
combined_images_test_flat = combined_images_test.reshape(combined_images_test.shape[0], -1)

# 데이터셋 크기 출력
print(f'Training data shape: {combined_images_train_flat.shape}')
print(f'Test data shape: {combined_images_test_flat.shape}')

# 데이터셋 샘플 확인
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(combined_images_train[i].reshape(28, 56), cmap='gray')
    plt.axis('off')
    plt.title(f'{combined_labels_train[i]:02d}')
plt.show()

# 데이터셋을 PyTorch Tensor로 변환 및 DataLoader 생성
X_train_tensor = torch.tensor(combined_images_train_flat, dtype=torch.float32)
y_train_tensor = torch.tensor(combined_labels_train, dtype=torch.long)
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

X_test_tensor = torch.tensor(combined_images_test_flat, dtype=torch.float32)
y_test_tensor = torch.tensor(combined_labels_test, dtype=torch.long)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# VQ-VAE 모델 정의
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        flatten = x.view(-1, self.embedding_dim)
        distances = (torch.sum(flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flatten, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view_as(x)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(torch.bincount(encoding_indices.flatten().to(torch.long)) / encoding_indices.numel())
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class VQVAE(nn.Module):
    def __init__(self, input_dim, num_latents, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.num_latents = num_latents
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_latents * embedding_dim)
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Linear(num_latents * embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = z_e.view(-1, self.num_latents, self.embedding_dim)
        quantized_list = []
        vq_loss = 0
        perplexity = 0
        for i in range(self.num_latents):
            quantized, loss, p = self.quantizer(z_e[:, i, :])
            quantized_list.append(quantized)
            vq_loss += loss
            perplexity += p
        z_q = torch.stack(quantized_list, dim=1).view(-1, self.num_latents * self.embedding_dim)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss / self.num_latents, perplexity / self.num_latents

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 1568
num_latents = 49
embedding_dim = 32
num_embeddings = 1024
commitment_cost = 0.25
vqvae = VQVAE(input_dim, num_latents, embedding_dim, num_embeddings, commitment_cost).to(device)

# VQ-VAE 모델 학습
vqvae_model_path = './RNN_data/VQVAE.pth'
os.makedirs('./RNN_data', exist_ok=True)

vqvae_loaded = False
if os.path.exists(vqvae_model_path):
    choice = input("학습된 VQ-VAE 모델이 있습니다. 로드하시겠습니까? (Y/N): ").strip().upper()
else:
    choice = "N"

if choice == "Y":
    vqvae.load_state_dict(torch.load(vqvae_model_path))
    vqvae_loaded = True
    print("VQ-VAE 모델이 로드되었습니다.")
else:
    optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)
    num_epochs = 20

    vqvae_losses = []
    for epoch in range(num_epochs):
        vqvae.train()
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            recon_images, vq_loss, _ = vqvae(images)
            recon_loss = F.mse_loss(recon_images, images)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        vqvae_losses.append(train_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], VQ-VAE Loss: {train_loss/len(train_loader):.4f}')

    # VQ-VAE 모델 저장
    torch.save(vqvae.state_dict(), vqvae_model_path)

# 코드북 기반의 표현 생성
vqvae.eval()

# 학습 데이터셋에서 인코더를 통해 잠재 벡터를 얻는 함수
def get_latent_vectors(loader, model, device):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            z_e = model.encoder(images)
            latent_vectors.append(z_e.cpu().numpy())
    return np.vstack(latent_vectors)

# 학습 데이터셋에서 잠재 벡터를 얻음
train_latent_vectors = get_latent_vectors(train_loader, vqvae, device)

# 테스트 데이터셋에서 잠재 벡터를 얻음
test_latent_vectors = get_latent_vectors(test_loader, vqvae, device)

# KMeans 클러스터링
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
train_clusters = kmeans.fit_predict(train_latent_vectors)
test_clusters = kmeans.predict(test_latent_vectors)

# 각 클러스터의 대표 라벨 설정
cluster_labels = np.zeros(n_clusters)
for i in range(n_clusters):
    labels_in_cluster = y_train_tensor[train_clusters == i]
    if len(labels_in_cluster) > 0:
        cluster_labels[i] = np.bincount(labels_in_cluster).argmax()

# 테스트 셋의 예측 라벨
test_pred_labels = cluster_labels[test_clusters]

# 정확도 계산
accuracy = accuracy_score(y_test_tensor, test_pred_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 결과 저장
results_df = pd.DataFrame({
    'True Label': y_test_tensor,
    'Predicted Label': test_pred_labels
})
results_df.to_csv(os.path.join(graphs_dir, 'test_results.csv'), index=False)

print("데이터와 라벨 저장 완료")