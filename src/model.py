import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchinfo import summary

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vitb16":
        """ Vision Transformer B16
        """
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
        model_ft = vit_b_16(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ResNet50
    print("=" * 80)
    print("ResNet50 Architecture")
    print("=" * 80)
    resnet_model, _ = initialize_model("resnet", num_classes=2, feature_extract=False, use_pretrained=False)
    resnet_model = resnet_model.to(device)
    summary(resnet_model, input_size=(1, 3, 224, 224), device=device, verbose=1)

    # Vision Transformer B16
    print("\n" + "=" * 80)
    print("Vision Transformer B16 Architecture")
    print("=" * 80)
    vitb16_model, _ = initialize_model("vitb16", num_classes=2, feature_extract=False, use_pretrained=False)
    vitb16_model = vitb16_model.to(device)
    summary(vitb16_model, input_size=(1, 3, 224, 224), device=device, verbose=1)
