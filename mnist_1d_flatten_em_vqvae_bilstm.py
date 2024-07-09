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

# 데이터 디렉토리 설정
data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 데이터 다운로드 여부를 확인
download = not os.path.exists(os.path.join(data_dir, 'MNIST/raw'))

# MNIST 데이터 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

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
        avg_probs = torch.mean(torch.bincount(encoding_indices.flatten()) / encoding_indices.numel())
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

# 학습된 VQ-VAE를 이용한 분류기 정의
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # bidirectional이므로 hidden_dim * 2

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # (batch_size, seq_len, input_dim)
        h0 = torch.zeros(num_layers * 2, x.size(0), hidden_dim).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(num_layers * 2, x.size(0), hidden_dim).to(x.device)  # 2 for bidirectional
        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 최종 시간 스텝의 출력을 사용
        return out

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 1568
num_latents = 49
num_embeddings_list = [256, 512, 1024]  # 실험할 num_embeddings 값 리스트
embedding_dim_list = [16, 32, 64]  # 실험할 embedding_dim 값 리스트
commitment_cost = 0.25

# 실험 반복
for num_embeddings in num_embeddings_list:
    for embedding_dim in embedding_dim_list:
        # 실험 결과 저장 디렉토리 설정
        experiment_dir = f'VQVAE_{embedding_dim}_{num_embeddings}'
        os.makedirs(experiment_dir, exist_ok=True)

        # VQ-VAE 모델 설정
        vqvae = VQVAE(input_dim, num_latents, embedding_dim, num_embeddings, commitment_cost).to(device)

        # VQ-VAE 모델 학습
        optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)
        num_epochs = 10

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
            print(f'Experiment {experiment_dir} - Epoch [{epoch+1}/{num_epochs}], VQ-VAE Loss: {train_loss/len(train_loader):.4f}')

        # VQ-VAE 모델 저장
        torch.save(vqvae.state_dict(), os.path.join(experiment_dir, 'VQVAE.pth'))

        # VQ-VAE 손실 그래프 그리기
        plt.figure()
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, vqvae_losses, label='VQ-VAE Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('VQ-VAE Loss', color='blue')
        plt.tick_params(axis='y', labelcolor='blue')
        plt.title('VQ-VAE Training Loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(experiment_dir, 'vqvae_loss.png'))
        plt.close()

        # BiLSTM 분류기 설정
        hidden_dim = 128
        num_layers = 2
        num_classes = 100

        classifier = BiLSTMClassifier(embedding_dim, hidden_dim, num_layers, num_classes).to(device)

        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        classification_criterion = nn.CrossEntropyLoss()

        # BiLSTM 학습
        rnn_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            classifier.train()
            epoch_loss = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    z_e = vqvae.encoder(images)
                    z_e = z_e.view(-1, num_latents, embedding_dim)
                    quantized_list = []
                    for i in range(num_latents):
                        quantized, _, _ = vqvae.quantizer(z_e[:, i, :])
                        quantized_list.append(quantized)
                    z_q = torch.stack(quantized_list, dim=1)

                optimizer.zero_grad()
                outputs = classifier(z_q)
                loss = classification_criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            rnn_losses.append(epoch_loss / len(train_loader))
            print(f'Experiment {experiment_dir} - Epoch [{epoch+1}/{num_epochs}], BiLSTM Loss: {epoch_loss/len(train_loader):.4f}')

            # Evaluate on the test set
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    z_e = vqvae.encoder(images)
                    z_e = z_e.view(-1, num_latents, embedding_dim)
                    quantized_list = []
                    for i in range(num_latents):
                        quantized, _, _ = vqvae.quantizer(z_e[:, i, :])
                        quantized_list.append(quantized)
                    z_q = torch.stack(quantized_list, dim=1)
                    outputs = classifier(z_q)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
            print(f'Experiment {experiment_dir} - Test Accuracy: {accuracy:.2f}%')

        # BiLSTM 모델 저장
        torch.save(classifier.state_dict(), os.path.join(experiment_dir, 'BiLSTM.pth'))

        # BiLSTM 손실 및 정확도 그래프 그리기
        plt.figure()
        plt.plot(epochs, rnn_losses, label='BiLSTM Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('BiLSTM Loss', color='blue')
        plt.tick_params(axis='y', labelcolor='blue')
        plt.title('BiLSTM Training Loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(experiment_dir, 'bilstm_loss.png'))
        plt.close()

        plt.figure()
        plt.plot(epochs, test_accuracies, label='Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)', color='red')
        plt.tick_params(axis='y', labelcolor='red')
        plt.title('BiLSTM Test Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(experiment_dir, 'bilstm_accuracy.png'))
        plt.close()

        # 손실 및 정확도 표로 저장
        results_df = pd.DataFrame({
            'Epoch': range(1, num_epochs + 1),
            'VQ-VAE Loss': vqvae_losses,
            'BiLSTM Loss': rnn_losses,
            'Accuracy': test_accuracies
        })

        results_df.to_csv(os.path.join(experiment_dir, 'loss_accuracy.csv'), index=False)

        # 데이터와 라벨 저장
        np.save(os.path.join(experiment_dir, 'X_train.npy'), combined_images_train_flat)
        np.save(os.path.join(experiment_dir, 'y_train.npy'), combined_labels_train)
        np.save(os.path.join(experiment_dir, 'X_test.npy'), combined_images_test_flat)
        np.save(os.path.join(experiment_dir, 'y_test.npy'), combined_labels_test)

        print(f'Experiment {experiment_dir} - 데이터와 라벨 저장 완료')
