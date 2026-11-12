"""Explicit ImageNet backbones; no changes to the lightweight PGCA graph."""
from query_model import HERE
import torch
from torch import nn
from torchvision.models import resnet50, densenet121, ResNet50_Weights, DenseNet121_Weights
from experiment_models import IndependentOrdinalHead

torch.hub.set_dir(str(HERE / "pretrained_cache"))
WEIGHTS = {"ResNet50-I": ResNet50_Weights.IMAGENET1K_V2,
           "DenseNet121-I": DenseNet121_Weights.IMAGENET1K_V1,
           "ResNet50-simple-IFP": ResNet50_Weights.IMAGENET1K_V2}


class PretrainedFusion(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.multimodal = name == "ResNet50-simple-IFP"
        if name.startswith("ResNet50"):
            self.backbone = resnet50(weights=WEIGHTS[name], progress=False)
            self.backbone.fc = nn.Identity()
            dimension = 2048
        else:
            self.backbone = densenet121(weights=WEIGHTS[name], progress=False)
            self.backbone.classifier = nn.Identity()
            dimension = 1024
        self.image_projection = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 256))
        if self.multimodal:
            self.feature_encoder = nn.Sequential(nn.Linear(36, 256), nn.ReLU(), nn.Dropout(.2), nn.Linear(256, 256))
            self.prompt_proj = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256))
        self.fusion = nn.Sequential(nn.Linear(768 if self.multimodal else 256, 256), nn.ReLU(), nn.Dropout(.2), nn.Linear(256, 256), nn.ReLU())
        self.head = IndependentOrdinalHead()
        self.register_buffer("imagenet_mean", torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def forward(self, image, features, phrase_emb, y=None):
        rgb = (image.repeat(1, 3, 1, 1) - self.imagenet_mean) / self.imagenet_std
        parts = [self.image_projection(self.backbone(rgb))]
        if self.multimodal:
            parts.extend([self.prompt_proj(phrase_emb), self.feature_encoder(features)])
        return self.head(self.fusion(torch.cat(parts, 1)), y)
