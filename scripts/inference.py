import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import argparse

class PhysicsModel(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, x):
        x = x.squeeze(1)
        inactive_mask = (x < self.threshold).float()
        inactive_pct = inactive_mask.mean(dim=(1, 2))
        pred = 1.5752 * inactive_pct - 0.6471
        return pred.unsqueeze(1), inactive_pct.unsqueeze(1)

class ModifiedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        
        # Modify first layer for grayscale
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.sum(dim=1, keepdim=True)
        
        # Rest of ResNet
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, None

class EnsembleModel(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.physics_model = PhysicsModel(threshold)
        self.resnet_model = ModifiedResNet18()
        self.meta = nn.Linear(2, 1)
        
    def forward(self, x):
        pred_physics, _ = self.physics_model(x)
        pred_resnet, _ = self.resnet_model(x)
        combined = torch.cat([pred_physics, pred_resnet], dim=1)
        return self.meta(combined)

def run_inference(args):
    """Run inference on a single image"""
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
        
        # Load and prepare model
        model = EnsembleModel(args.threshold)
        model.load_state_dict(torch.load(args.model))
        model.to(device).eval()
        
        # Load and preprocess image
        img = Image.open(args.image).convert('L')
        img = img.resize((args.size, args.size))
        img = torch.FloatTensor(np.array(img)) / 255.0
        img = img.unsqueeze(0).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            pred_physics, _ = model.physics_model(img)
            pred_resnet, _ = model.resnet_model(img)
            final_pred = model(img)
            
            # Convert to numpy
            pred_physics = pred_physics.cpu().numpy()[0][0]
            pred_resnet = pred_resnet.cpu().numpy()[0][0]
            final_pred = final_pred.cpu().numpy()[0][0]
        
        # Calculate power loss
        power_loss = 1 - final_pred
        
        # Print results
        print(f"\nResults for {os.path.basename(args.image)}:")
        print(f"   Physics Model Prediction: {pred_physics:.4f}")
        print(f"   ResNet Model Prediction: {pred_resnet:.4f}")
        print(f"   Ensemble Prediction: {final_pred:.4f}")
        print(f"   Power Loss: {power_loss:.4f}")
        print(f"   Relative Power: {final_pred:.4f}")
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        result_file = os.path.join(args.output, 
                                 f"{os.path.splitext(os.path.basename(args.image))[0]}_results.txt")
        
        with open(result_file, 'w') as f:
            f.write(f"Physics Model Prediction: {pred_physics:.4f}\n")
            f.write(f"ResNet Model Prediction: {pred_resnet:.4f}\n")
            f.write(f"Ensemble Prediction: {final_pred:.4f}\n")
            f.write(f"Power Loss: {power_loss:.4f}\n")
            f.write(f"Relative Power: {final_pred:.4f}\n")
            
        return {
            "physics_pred": pred_physics,
            "resnet_pred": pred_resnet,
            "ensemble_pred": final_pred,
            "power_loss": power_loss,
            "relative_power": final_pred
        }
            
    except Exception as e:
        print(f"\nError processing {os.path.basename(args.image)}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Solar Panel EL Image Analysis")
    parser.add_argument('--image', required=True, help="Path to input image")
    parser.add_argument('--model', required=True, help="Path to model weights")
    parser.add_argument('--output', required=True, help="Output directory")
    parser.add_argument('--gpu', type=lambda x: x.lower() == 'true', required=True, help="Use GPU")
    parser.add_argument('--size', type=int, required=True, help="Image size")
    parser.add_argument('--threshold', type=float, required=True, help="Threshold value")
    
    args = parser.parse_args()
    run_inference(args)

if __name__ == "__main__":
    main() 
