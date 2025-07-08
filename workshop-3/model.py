import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import BertModel

# Multimodal Transformer Model copied shamelessly from the notebook
class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Image encoder
        self.img_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.img_encoder.fc = nn.Identity()
        self.img_proj = nn.Linear(512, 768)  # Project to text dimension
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Fusion transformer
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True),
            num_layers=2
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, img, text):
        # Image features
        img_feats = self.img_encoder(img)
        img_feats = self.img_proj(img_feats).unsqueeze(1)  # [batch, 1, 768]
        
        # Text features
        text_feats = self.text_encoder(
            input_ids=text['input_ids'],
            attention_mask=text['attention_mask']
        ).last_hidden_state
        
        # Combine features
        combined = torch.cat([img_feats, text_feats], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        return self.classifier(fused[:, 0, :])  # Use image token
    