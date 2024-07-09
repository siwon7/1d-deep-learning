import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# IMDb 데이터셋 로드
dataset = load_dataset('imdb')

# IMDb 데이터셋 전처리
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CombinedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, sequence_length):
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.data = self.prepare_data(texts, tokenizer)

    def prepare_data(self, texts, tokenizer):
        combined_text = []
        for text in texts:
            combined_text.extend(tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=self.sequence_length))
        # 일정 길이로 자르기
        chunks = [combined_text[i:i + self.sequence_length] for i in range(0, len(combined_text), self.sequence_length)]
        return chunks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return torch.tensor(chunk, dtype=torch.long)

sequence_length = 512  # 입력 길이 설정

# 훈련 데이터에서 1000개만 사용
train_texts = [x['text'] for x in dataset['train'].select(range(1000))]

train_dataset = CombinedTextDataset(train_texts, tokenizer, sequence_length)

def collate_fn(batch):
    batch_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    attention_mask = (batch_padded != 0).long()
    return batch_padded, attention_mask

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

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
        avg_probs = torch.mean(torch.bincount(encoding_indices.flatten(), minlength=self.num_embeddings).float() / encoding_indices.numel())
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, input_dim, num_latents, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.num_latents = num_latents
        self.embedding_dim = embedding_dim

        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_latents * embedding_dim)
        )
        
        # VectorQuantizer 인스턴스 생성
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.Linear(num_latents * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        # 입력 데이터를 float 타입으로 변환
        x = x.float()
        # 인코더를 통해 잠재 벡터 생성
        z_e = self.encoder(x)
        z_e = z_e.view(-1, self.num_latents, self.embedding_dim)

        # 잠재 벡터를 퀀타이즈
        quantized_list = []
        vq_loss = 0
        perplexity = 0
        encoding_indices_list = []
        for i in range(self.num_latents):
            quantized, loss, p, encoding_indices = self.quantizer(z_e[:, i, :])
            quantized_list.append(quantized)
            vq_loss += loss
            perplexity += p
            encoding_indices_list.append(encoding_indices)

        # 퀀타이즈된 벡터 결합 및 디코딩
        z_q = torch.stack(quantized_list, dim=1).view(-1, self.num_latents * self.embedding_dim)
        encoding_indices = torch.stack(encoding_indices_list, dim=1)
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss / self.num_latents, perplexity / self.num_latents, encoding_indices

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = sequence_length  # 입력 길이는 정해진 문장 길이
num_latents = 32  # 잠재 공간의 크기 설정
embedding_dim = 16
num_embeddings = 256
commitment_cost = 0.25

vqvae = VQVAE(input_dim, num_latents, embedding_dim, num_embeddings, commitment_cost).to(device)

# VQ-VAE 모델 학습
vqvae_model_path = 'vqvae_sentiment.pth'
if os.path.exists(vqvae_model_path):
    choice = input("학습된 VQ-VAE 모델이 있습니다. 로드하시겠습니까? (Y/N): ").strip().upper()
else:
    choice = "N"

if choice == "Y":
    vqvae.load_state_dict(torch.load(vqvae_model_path))
    print("VQ-VAE 모델이 로드되었습니다.")
else:
    optimizer = optim.Adam(vqvae.parameters(), lr=1e-5)  # 학습률을 낮춤
    num_epochs = 10

    vqvae_losses = []
    for epoch in range(num_epochs):
        vqvae.train()
        train_loss = 0
        for texts, attention_mask in train_loader:
            texts = texts.to(device).float()  # Float 타입으로 변환

            optimizer.zero_grad()
            recon_texts, vq_loss, _, _ = vqvae(texts)
            recon_loss = F.mse_loss(recon_texts, texts)  # 수정된 부분
            loss = recon_loss + vq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)  # 그라디언트 클리핑
            optimizer.step()
            train_loss += loss.item()

        vqvae_losses.append(train_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], VQ-VAE Loss: {train_loss/len(train_loader):.4f}')

    # VQ-VAE 모델 저장
    torch.save(vqvae.state_dict(), vqvae_model_path)

# 코드북 내용 확인
vqvae.eval()
with torch.no_grad():
    texts, _ = next(iter(train_loader))
    texts = texts.to(device).float()  # Float 타입으로 변환
    _, _, _, encoding_indices = vqvae(texts)

# 코드북 내용 출력
codebook = [[] for _ in range(num_embeddings)]
for i in range(encoding_indices.shape[0]):
    for j in range(encoding_indices.shape[1]):
        codebook[encoding_indices[i, j].item()].append(texts[i, j].item())

for i in range(num_embeddings):
    if codebook[i]:
        words = [tokenizer.decode([int(idx)]) for idx in codebook[i]]
        print(f"Codebook index {i}: {' '.join(words)}")
