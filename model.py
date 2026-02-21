import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2

class MCDropoutResNet(nn.Module):
    """
    ResNet50 extended with Monte Carlo Dropout and Grad-CAM hooks
    for uncertainty estimation and explainability.
    """
    def __init__(self, num_classes=1, dropout_prob=0.5):
        super(MCDropoutResNet, self).__init__()
        # Load pretrained ResNet50
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.base_model.fc.in_features
        
        # Replace the fully connected layer with a dropout classifier
        # Outputting 1 logit for Binary Classification (Disease vs Normal)
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Hook storage for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Register hooks on the last convolutional layer
        target_layer = self.base_model.layer4[-1]
        target_layer.register_forward_hook(self.save_activation)
        # Using register_full_backward_hook to avoid PyTorch warnings
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.base_model(x)
        
    def generate_gradcam(self, input_tensor):
        """
        Generates a Grad-CAM heatmap for the given input tensor.
        NOTE: Must run in eval mode or train mode depending on whether 
        we want deterministic CAM or MC CAM. Generally done in eval mode.
        """
        self.eval() # Ensure deterministic forward pass for CAM
        
        # Forward pass to get activations
        output = self.forward(input_tensor)
        
        # Backward pass from the specific class output
        self.base_model.zero_grad()
        output[0, 0].backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the channels by the gradients
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # Go through ReLU to keep only positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize between 0-1 and resize to original spatial size (224x224)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        return cam, torch.sigmoid(output).item()

def predict_with_uncertainty(model, input_tensor, num_samples=15):
    """
    Applies Monte Carlo Dropout Inference.
    Runs multiple forward passes with Dropout enabled.
    Returns: Mean Probability, Confidence, Uncertainty
    """
    model.eval()  # Crucial: Keep BatchNorm in eval mode
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train() # Force only dropout layers to train mode
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Output is logits; apply sigmoid
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
            predictions.append(prob)
            
    predictions = np.array(predictions)
    
    # Mathematical Model execution
    mean_prob = np.mean(predictions)
    variance = np.var(predictions)
    
    # The maximum variance for a probability bounded [0,1] is 0.25 (when p=0.5).
    # We can normalize uncertainty to a 0-1 scale by multiplying variance by 4.
    uncertainty = min(variance * 4.0, 1.0) 
    
    # As defined by requirement: Confidence = 1 - Uncertainty
    confidence = max(1.0 - uncertainty, 0.0) 
    
    return mean_prob, confidence, uncertainty
