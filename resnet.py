from torchvision.models import resnet50, ResNet50_Weights
from torchinfo import summary
# Using pretrained weights:
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

summary(model, input_size=(1, 3, 613, 293), depth=100)
