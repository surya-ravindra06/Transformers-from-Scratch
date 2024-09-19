import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nltk
import re
import torch.optim as optim
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = './train.en'

# Read the dataset
with open(dataset_path, 'r') as f:
    data = f.read()

data = data.lower()
data = re.sub(r'[^a-zA-Z0-9\n]', ' ', data)

# Split the dataset into sentences
sentences = data.split('\n')

# tokenize the sentences
tokens = []
for sentence in sentences:
    tokens.append(nltk.word_tokenize(sentence))

print(tokens[0])
eng_len = len(tokens)

# Create a vocabulary

word2idx = {}
idx2word = {}

word2idx['<pad>'] = 0
word2idx['<unk>'] = 1
idx2word[0] = '<pad>'
idx2word[1] = '<unk>'

for sentence in tokens:
    for word in sentence:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            idx2word[len(idx2word)] = word

vocab_size = len(word2idx)

dataset_path = './train.fr'

# Read the dataset
with open(dataset_path, 'r') as f:
    data = f.read()

data = data.lower()
data = re.sub(r'\.', '', data)

# Split the dataset into sentences
french_sentences = data.split('\n')

print(french_sentences[1])

# tokenize the sentences
french_tokens = []
for sentence in french_sentences:
    temp_token_list = []
    for word in sentence.split(' '):
        temp_token_list.append(word)
    french_tokens.append(temp_token_list)

print(french_tokens[1])
fr_len = len(french_tokens)

# Create a vocabulary

fr_word2idx = {}
fr_idx2word = {}

fr_word2idx['<pad>'] = 0
fr_word2idx['<unk>'] = 1
fr_idx2word[0] = '<pad>'
fr_idx2word[1] = '<unk>'

for sentence in french_tokens:
    for word in sentence:
        if word not in fr_word2idx:
            fr_word2idx[word] = len(fr_word2idx)
            fr_idx2word[len(fr_idx2word)] = word

vocab_size = len(fr_word2idx)

# pad the sentences to the length 50 for both english and french
max_len = 500
for i in range(len(tokens)):
    if len(tokens[i]) < max_len:
        tokens[i] = tokens[i] + ['<pad>'] * (max_len - len(tokens[i]))
    else:
        tokens[i] = tokens[i][:max_len]

for i in range(len(french_tokens)):
    if len(french_tokens[i]) < max_len:
        french_tokens[i] = french_tokens[i] + ['<pad>'] * (max_len - len(french_tokens[i]))
    else:
        french_tokens[i] = french_tokens[i][:max_len]

print(tokens[0])

class Dataset(Dataset):
    def __init__(self, english_sentences, french_sentences, eng_len, fr_len):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        self.eng_len = eng_len
        self.fr_len = fr_len
    
    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, idx):
        eng_sentence = self.english_sentences[idx]
        fr_sentence = self.french_sentences[idx]
        eng_sentence = [word2idx[word] for word in eng_sentence]
        fr_sentence = [fr_word2idx[word] for word in fr_sentence]
        return torch.LongTensor(eng_sentence), torch.LongTensor(fr_sentence)
    
train_iter = Dataset(tokens, french_tokens, eng_len, fr_len)
train_loader = DataLoader(train_iter, batch_size=16, shuffle=True)

for i in range(5):
    print(train_iter[i])
    print(train_iter[i][1].shape)

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def forward(self, x):
        return self.embeddings(x)

max_len = 500 # This is based on my GPUs memory

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super(PositionalEncoding, self).__init__()
        positional_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))
        
    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1)]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_k = embedding_dim // num_heads
        
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embedding_dim // self.num_heads)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, embedding_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output 

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(embedding_dim, d_ff)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model).to(device)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model).to(device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length).to(device)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]).to(device)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]).to(device)

        self.fc = nn.Linear(d_model, tgt_vocab_size).to(device)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

en_vocab_size = len(word2idx)
fr_vocab_size = len(fr_idx2word)
embedding_dim = 300
num_heads = 3
num_layers = 2
ffn_dim = 1024
max_len = 500
dropout = 0.1

model = Transformer(en_vocab_size, fr_vocab_size, embedding_dim, num_heads, num_layers, ffn_dim, max_len, dropout).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)

# calculating the BLEU score
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(pred, truth):
    #print(pred.shape, truth.shape)
    #print(pred, truth)
    pred = pred.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()
    # convert to 1d
    pred = pred.flatten()
    truth = truth.flatten()
    # convert to int
    preds = []
    truths = []
    for item in pred:
        #print(item)
        preds.append(int(item))
    for item in truth:
        truths.append(int(item))
    #print(preds, truths)
    chencherry = SmoothingFunction()
    bleu_score = sentence_bleu([truths], preds, smoothing_function=chencherry.method1)
    return bleu_score

epochs = 25

model.train()
model = model.to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="transformer-translation",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "Transformer",
    "dataset": "English to French",
    "epochs": 25,
    }
)

val_iter = Dataset(tokens, french_tokens, eng_len, fr_len)
val_loader = DataLoader(val_iter, batch_size=1, shuffle=True)


for epoch in range(epochs):
    for i, (src, tgt) in enumerate(train_loader):
        optimizer.zero_grad()
        src = src.to(device)
        tgt = tgt.to(device)
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        # bleu_score = calculate_bleu(output.argmax(dim=-1), tgt[:, 1:])
        # add a loop for validation set also
        # for i, (src, tgt) in enumerate(val_loader):
        #     src = src.to(device)
        #     tgt = tgt.to(device)
        #     output = model(src, tgt[:, :-1])
        #     loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        #     val_bleu_score = calculate_bleu(output.argmax(dim=-1), tgt[:, 1:])
        #     wandb.log({"val_bleu_score": val_bleu_score})
        #     print('Device: {}, Epoch: {}/{}, Loss: {}, Val Bleu Score: {}'.format(device, epoch, epochs, loss.item(), val_bleu_score))
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        wandb.log({"loss": loss})
        print('Device: {}, Epoch: {}/{}, Loss: {}'.format(device, epoch, epochs, loss.item()))

torch.save(model, 'transformer.pth')
torch.save(word2idx, 'word2idx.pth')
torch.save(fr_word2idx, 'fr_word2idx.pth')
torch.save(idx2word, 'idx2word.pth')
torch.save(fr_idx2word, 'fr_idx2word.pth')

# calculate the BLEU score
model.eval()
model = model.to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter = Dataset(tokens, french_tokens, eng_len, fr_len)
train_loader = DataLoader(train_iter, batch_size=1, shuffle=True)

bleu_scores = []
sum_bleu_scores = 0
for i, (src, tgt) in enumerate(train_loader):
    src = src.to(device)
    tgt = tgt.to(device)
    output = model(src, tgt[:, :-1])
    bleu_score = calculate_bleu(output.argmax(dim=-1), tgt[:, 1:])
    bleu_scores.append(bleu_score)
    sum_bleu_scores += bleu_score
    print('Device: {}, Bleu score: {}'.format(device, bleu_score))

print('Average Bleu score: {}'.format(sum_bleu_scores / len(bleu_scores)))





