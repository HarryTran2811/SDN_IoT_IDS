import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig, TrainingArguments, Trainer
from torchvision import transforms
from datasets import load_dataset


class ViTModelTrainer:
    def __init__(self, num_labels=10, image_size=224, patch_size=16, hidden_size=768,
                 num_attention_heads=12, num_hidden_layers=12, intermediate_size=3072,
                 classifier_dropout=0.1, learning_rate=3e-5, weight_decay=0.01,
                 batch_size=32, num_epochs=10):
        self.config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_labels=num_labels,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            classifier_dropout=classifier_dropout
        )
        self.model = ViTForImageClassification(self.config)

        self.training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_strategy="epoch",
            logging_dir='./logs'
        )

        # Chuẩn hóa theo tập ImageNet

        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def load_and_preprocess_data(self, dataset_name):

    def train(self):


if __name__ == "__main__":
