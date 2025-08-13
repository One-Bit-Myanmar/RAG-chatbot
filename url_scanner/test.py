import math
import torch
import torch.nn as nn
from transformers import BertTokenizer

# =========================
# Device configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# =========================
# Transformer URL Classifier
# =========================
class URLTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=2, num_layers=2, max_seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),       
        )
            
    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=(mask == 0))
        else:
            x = self.transformer_encoder(x)

        pooled = x.mean(dim=1)  # Mean pooling over sequence
        out = self.classifier(pooled)  # Shape: (batch, 1)
        return out

# =========================
# Load Model and Tokenizer
# =========================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = URLTransformerClassifier(
    vocab_size=tokenizer.vocab_size,
    embed_dim=256,
    num_heads=2,
    num_layers=2,
    max_seq_len=64
).to(device)

# Load saved weights
model.load_state_dict(torch.load(
    "url_scanner/save_urlScanner/transformer_ver_1.2-2025-08-09.pth",
    map_location=device
))

model.eval()

# =========================
# Predict Function
# =========================
def predict_url(url):
    encoding = tokenizer(
        url,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask).squeeze(1)
        prob = torch.sigmoid(output).item()
        threshold = 0.7  # or 0.8
        pred_label = "malicious" if prob < threshold else "safe"


    return pred_label, prob

# =========================
# Test Example
# =========================
test_url = "https://www.google.com"
label, probability = predict_url(test_url)

print(f"URL: {test_url}")
print(f"Prediction: {label} (probability: {probability:.4f})")
