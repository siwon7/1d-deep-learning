import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split

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
def create_combined_images(x_samples, num_samples_per_class=100):
    combined_images = []
    combined_labels = []
    for i in range(100):
        for _ in range(num_samples_per_class):
            num1 = i // 10
            num2 = i % 10
            img1 = x_samples[num1 * num_samples + np.random.randint(num_samples)]
            img2 = x_samples[num2 * num_samples + np.random.randint(num_samples)]
            combined_img = np.hstack((img1[0], img2[0]))
            combined_label = num1 * 10 + num2
            combined_images.append(combined_img)
            combined_labels.append(combined_label)
    return np.array(combined_images), np.array(combined_labels)

# 10,000개의 training dataset 생성
combined_images, combined_labels = create_combined_images(x_train_samples, num_samples_per_class=100)

# 이미지 평탄화
combined_images_flat = combined_images.reshape(combined_images.shape[0], -1)

# 학습 및 테스트 데이터셋 분할 (총 12,000개의 데이터 생성하여 10,000개 학습, 2,000개 테스트)
X_train, X_test, y_train, y_test = train_test_split(combined_images_flat, combined_labels, test_size=0.1667, random_state=42)

# 데이터셋 크기 출력
print(f'Training data shape: {X_train.shape}')
print(f'Test data shape: {X_test.shape}')

# 데이터셋 샘플 확인
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(combined_images[i], cmap='gray')
    plt.axis('off')
    plt.title(f'{combined_labels[i]:02d}')
plt.show()

# 데이터셋을 PyTorch Tensor로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader 생성
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 개선된 GRU 기반 모델 정의
class ImprovedGRU1D(nn.Module):
    def __init__(self):
        super(ImprovedGRU1D, self).__init__()
        self.gru = nn.GRU(input_size=56, hidden_size=256, num_layers=3, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 100)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.view(-1, 28, 56)  # (batch_size, seq_len, input_size)
        _, h_n = self.gru(x)
        x = h_n[-1]
        x = torch.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = ImprovedGRU1D()

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 그래프를 저장할 폴더 생성
os.makedirs('GRU_graphs', exist_ok=True)

# 학습 시작 시간 기록
start_time = time.time()

# 모델 학습
num_epochs = 50
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')
    
    # Evaluate on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f'Test Accuracy: {accuracy:.2f}%')

# 학습 종료 시간 기록
end_time = time.time()

# 전체 학습 시간 계산 및 출력
total_time = end_time - start_time
print(f'Total Training Time: {total_time:.2f} seconds')

# 학습 종료 후 그래프 그리기
plt.figure()
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, label='Loss', color='blue')
plt.xlabel('Epochs')

plt.ylabel('Loss', color='blue')
plt.tick_params(axis='y', labelcolor='blue')

ax2 = plt.gca().twinx()
ax2.plot(epochs, test_accuracies, label='Accuracy', color='red')
ax2.set_ylabel('Accuracy (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Training Loss and Test Accuracy')
plt.legend(loc='upper right')
plt.savefig(f'GRU_graphs/loss_accuracy.png')
plt.close()

# 손실 및 정확도 표로 저장
results_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Loss': train_losses,
    'Accuracy': test_accuracies
})

results_df.to_csv('GRU_graphs/loss_accuracy.csv', index=False)

# 학습 시간 저장
time_df = pd.DataFrame({
    'Total Training Time (seconds)': [total_time]
})

time_df.to_csv('GRU_graphs/training_time.csv', index=False)

# 모델 평가 및 예측 저장
model.eval()
os.makedirs('GRU', exist_ok=True)
for i in range(100):
    os.makedirs(f'GRU/{i:02d}', exist_ok=True)

with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for j in range(images.size(0)):
            img = images[j].cpu().numpy().reshape(28, 56)
            label = predicted[j].item()
            plt.imsave(f'GRU/{label:02d}/{i*32+j}.png', img, cmap='gray')

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 데이터와 라벨 저장
os.makedirs('combined_data', exist_ok=True)
np.save('combined_data/X_train.npy', X_train_tensor.cpu().numpy())
np.save('combined_data/y_train.npy', y_train_tensor.cpu().numpy())
np.save('combined_data/X_test.npy', X_test_tensor.cpu().numpy())
np.save('combined_data/y_test.npy', y_test_tensor.cpu().numpy())

print("데이터와 라벨 저장 완료")
