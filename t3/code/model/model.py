import numpy as np

import torch 
import torch.nn as nn 

from monai.utils import set_determinism
import sys
import os



from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import BasicUNetDe

set_determinism(123)

# Set the data directory.
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Set the log directory.
logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_btcv/diffunet_transunet_datasettings/")
if not os.path.exists(logdir):
    os.makedirs(logdir)

model_save_path = os.path.join(logdir, "model")
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Training parameters for the legacy model smoke test.
max_epoch = 1000  # Keep the epoch count small for testing.
batch_size = 1
val_every = 50
env = "pytorch"  # Use single-GPU training.
num_gpus = 1

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class DiffUNet(nn.Module):
    def __init__(self, con_channels=2, model_size='lite', activation_checkpointing=False) -> None:
        """
        Initialize DiffUNet model.
        
        Args:
            con_channels: Number of input condition channels
            model_size: Model size, either 'lite' or 'large'
        """
        super().__init__()
        
        # Import MODEL_FEATURES from config
        from config import MODEL_FEATURES
        
        # Get features based on model size
        if model_size not in MODEL_FEATURES:
            raise ValueError(f"model_size must be 'lite' or 'large', got {model_size}")
        features = MODEL_FEATURES[model_size]
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = activation_checkpointing
        
        self.embed_model = BasicUNetEncoder(
            spatial_dims=2,           # 2D image
            in_channels=con_channels,
            out_channels=features[0],  # Use first feature as out_channels
            features=features,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            bias=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.model = BasicUNetDe(
            spatial_dims=2,           # 2D image
            in_channels=con_channels+1,
            out_channels=1,
            features=features,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
            norm=("instance", {"affine": True}),
            bias=True,
            use_cross_attention=True,
            activation_checkpointing=activation_checkpointing,
        )
        self.cfg_drop_prob = 0.25

        

    def forward(self, image=None, x=None, pred_type='denoise', step=None, embedding=None):
        """
        Args:
            image: Input image with shape [B, 2, H, W].
            x: Input noise with shape [B, 1, H, W].
            pred_type: Prediction type; defaults to "denoise".
            step: Time step.
            embedding: Optional embedding list. If omitted, compute it from image.
        """
        # Expand a scalar time step to the batch size.
        if step.dim() == 0:
            step = step.repeat(x.shape[0])
        
        if pred_type == "denoise":
            # Compute the embedding list when it is not supplied.
            embeddings = self.embed_model(image) if embedding is None else embedding

            # During training, mask all embeddings for selected samples.
            if self.training and hasattr(self, 'cfg_drop_prob'):
                # A true entry marks a sample to mask.
                mask = torch.rand(embeddings[0].shape[0], device=embeddings[0].device) < self.cfg_drop_prob
                if mask.any():
                    # Broadcast the batch mask across each embedding tensor.
                    mask_reshape = mask.view(-1, *[1]*(embeddings[0].dim()-1))
                    for i in range(len(embeddings)):
                        # Avoid in-place mutation by using masked_fill.
                        embeddings[i] = embeddings[i].masked_fill(mask_reshape, 0.0)

            # Convert a scalar time step to a tensor.
            if isinstance(step, int):
                step = torch.tensor([step]).to(x.device)
            elif isinstance(step, torch.Tensor) and step.dim() == 0:
                step = step.unsqueeze(0)
            
            # Return the denoised result using the embedding list.
            return self.model(x, t=step, image=image, embeddings=embeddings)

    def forward_with_cfg(self, image=None, x=None, step=None, embedding=None, cfg_scale=2.5):
        """
        Forward pass with classifier-free guidance (CFG).
        
        Args:
            image: Input image with shape [B, 2, H, W].
            x: Input noise with shape [B, 1, H, W].
            step: Time step.
            embedding: Optional embedding list. If omitted, compute it from image.
            cfg_scale: CFG scale combining conditional and unconditional predictions.
        
        Returns:
            Denoised result combined with CFG as follows:
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        """
        # Expand a scalar time step to the batch size.
        if step.dim() == 0:
            step = step.repeat(x.shape[0])
        
        # Obtain the conditional embedding list.
        if embedding is None:
            embedding = self.embed_model(image)
        cond_embedding = embedding

        # Match each conditional embedding with an all-zero tensor.
        uncond_embedding = [torch.zeros_like(e) for e in embedding]

        # Compute the unconditional prediction.
        pred_uncond = self.model(x, t=step, image=image, embeddings=uncond_embedding)
        # Compute the conditional prediction.
        pred_cond = self.model(x, t=step, image=image, embeddings=cond_embedding)

        # Combine predictions using the CFG formula.
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)



def test_diffunet():
    print("Testing the DiffUNet model...")
    try:
        # Create random input data.
        image = torch.randn(10, 2, 256, 256).to(device)  # Batch of ten two-channel images.
        noise = torch.randn(10, 1, 256, 256).to(device)  # Batch of ten one-channel noise maps.
        print("Input image shape:", image.shape)
        print("Input noise shape:", noise.shape)

        # Instantiate the model.
        model = DiffUNet().to(device)
        print("Model created and moved to device:", device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Lightweight model parameter count: {total_params:,}")
        from torchsummary import summary
        #summary(model, input_size=(2, 256, 256))
        # Run a forward pass.
        import time
        with torch.no_grad():
            start_time = time.time()
            output = model(image=image, x=noise, pred_type="denoise", step=torch.tensor([0]).to(device))
            print("Forward pass succeeded")
            end_time = time.time()
            print("time:",end_time-start_time)
            output = model.forward_with_cfg(image=image, x=noise, step=torch.tensor([0]).to(device))
            print("Forward pass succeeded")
            print("Output shape:", output.shape)
            print("Output maximum:", output.max().item())
            print("Output minimum:", output.min().item())
            print("Output mean:", output.mean().item())

        print("Test completed")
        return True
    except Exception as e:
        print("An error occurred during the test:", str(e))
        return False

if __name__ == "__main__":
    # Run the smoke test first.
    test_success = test_diffunet()
