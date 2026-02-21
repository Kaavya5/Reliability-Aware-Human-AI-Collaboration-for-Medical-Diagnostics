import torch
from torchvision import transforms
from PIL import Image

def get_transforms():
    """
    Returns the standard validation/inference transforms for a ResNet trained on ImageNet
    or a generic chest X-ray dataset.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def calculate_risk_metrics(probability: float, confidence: float, uncertainty: float, severity: float, safety_weight: float):
    """
    Core Mathematical Model:
    Risk = P × S
    Reliability = C × (1 - U)
    AdjustedRisk = Risk + (1 - Reliability) × W
    """
    risk = probability * severity
    reliability = confidence * (1.0 - uncertainty)
    adjusted_risk = risk + (1.0 - reliability) * safety_weight
    
    return risk, reliability, adjusted_risk

def determine_priority(adjusted_risk: float, t1_urgent: float = 3.5, t2_review: float = 2.0):
    """
    Decision Layer thresholds mapping:
    - AdjustedRisk >= T1 -> Urgent
    - AdjustedRisk >= T2 -> Review
    - Else -> Routine
    """
    if adjusted_risk >= t1_urgent:
        return "Urgent"
    elif adjusted_risk >= t2_review:
        return "Review"
    else:
        return "Routine"
