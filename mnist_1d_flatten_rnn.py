import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

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

# RNN 분류기 정의
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, input_dim) -> (batch_size, seq_len, input_dim)
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 최종 시간 스텝의 출력을 사용
        return out

input_dim = X_train_tensor.shape[1]
hidden_dim = 128
num_layers = 2
num_classes = 100

classifier = RNNClassifier(input_dim, hidden_dim, num_layers, num_classes)

# 손실 함수 및 옵티마이저 정의
classification_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# 그래프를 저장할 폴더 생성
os.makedirs('RNN_graphs', exist_ok=True)

# 모델 가중치를 저장할 폴더 생성
os.makedirs('model', exist_ok=True)

# 모델 가중치 파일 경로
model_path = 'model/RNN.pth'

# 학습된 모델이 있는지 확인하고 사용자에게 묻기
if os.path.exists(model_path):
    choice = input("학습된 모델이 있습니다. 재학습을 원하십니까? (Y/N): ").strip().upper()
else:
    choice = "Y"

if choice == "Y":
    # 학습 시작 시간 기록
    start_time = time.time()

    # 학습과 평가 코드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    num_epochs = 50
    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        classifier.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = classification_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

        # Evaluate on the test set
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = classifier(images)
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

    # 모델 가중치 저장
    torch.save(classifier.state_dict(), model_path)

    # 그래프 저장할 폴더 생성 (이미 생성된 경우 무시)
    os.makedirs('RNN_graphs', exist_ok=True)

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
    plt.savefig(f'RNN_graphs/loss_accuracy.png')
    plt.close()

    # 손실 및 정확도 표로 저장
    results_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Loss': train_losses,
        'Accuracy': test_accuracies
    })

    results_df.to_csv('RNN_graphs/loss_accuracy.csv', index=False)

    # 학습 시간 저장
    time_df = pd.DataFrame({
        'Total Training Time (seconds)': [total_time]
    })

    time_df.to_csv('RNN_graphs/training_time.csv', index=False)
else:
    # 모델 로드
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    print("모델이 로드되었습니다. 테스트를 시작합니다.")

# 테스트 분류 시간 측정
test_start_time = time.time()

with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_end_time = time.time()
test_total_time = test_end_time - test_start_time
print(f'Test Time: {test_total_time:.2f} seconds')

# 테스트 시간 저장
test_time_df = pd.DataFrame({
    'Total Test Time (seconds)': [test_total_time]
})

test_time_df.to_csv('RNN_graphs/test_time.csv', index=False)

# 모델 평가 및 예측 저장
classifier.eval()
os.makedirs('RNN', exist_ok=True)
for i in range(100):
    os.makedirs(f'RNN/{i:02d}', exist_ok=True)

with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for j in range(images.size(0)):
            img = images[j].cpu().numpy().reshape(28, 56)
            label = predicted[j].item()
            plt.imsave(f'RNN/{label:02d}/{i*32+j}.png', img, cmap='gray')

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 데이터와 라벨 저장
os.makedirs('combined_data', exist_ok=True)
np.save('combined_data/X_train.npy', combined_images_train_flat)
np.save('combined_data/y_train.npy', combined_labels_train)
np.save('combined_data/X_test.npy', combined_images_test_flat)
np.save('combined_data/y_test.npy', combined_labels_test)

print("데이터와 라벨 저장 완료")
