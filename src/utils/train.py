import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=8,
    classes=2
)
model.load_state_dict(torch.load("ftw-2class-full_unet-efficientnetb3_rgbnir_f2444768.pth", weights_only=True))