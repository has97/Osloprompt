
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Dict, Tuple
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import einops
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

# CLIP and Diffusion imports
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPVisionModel, CLIPImageProcessor
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionInstructPix2PixPipeline,
    DPMSolverMultistepScheduler
)
import warnings
warnings.filterwarnings('error', category=UserWarning, message='.*deterministic.*')

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
    # Additional deterministic settings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set default tensor type to avoid precision issues
    # torch.set_default_dtype(torch.float16)
seed_everything(42)

torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Initialize global components
device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = _Tokenizer()
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

class PromptProjector(nn.Module):
    def __init__(self, input_dim, project_dim, output_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, project_dim)
        self.hidden = nn.Sequential(
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(project_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.output_proj = nn.Linear(project_dim, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.hidden(x)  # Residual connection
        return self.output_proj(x)
    
def load_clip_to_cpu(cfg):
    """Load CLIP model to CPU for initialization."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def get_image_transforms():
    """Get standard image preprocessing transforms."""
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    return resize_transform, normalize


def disable_safety_checker(images, clip_input):
    """Disable safety checker for diffusion models."""
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


class TextEncoder(nn.Module):
    """CLIP Text Encoder ."""
    
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, feature=None):
        x = prompts.to(device) + self.positional_embedding.type(self.dtype).to(device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if feature is not None:
            feature = einops.repeat(feature, 'm n -> k m n', k=5)
            x[:5, :, :] = x[:5, :, :] + feature
            
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class StableDiffusion(nn.Module):
    """Stable Diffusion wrapper for image generation."""
    # model_id="runwayml/stable-diffusion-v1-5"
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        super().__init__()
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     model_id, 
        #     torch_dtype=torch.float16
        # ).to(device)
        torch.manual_seed(42)
    
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,  # Use float32 for better determinism
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only = True
            
        ).to(device)
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=False  # Disable for determinism
        )
        
        # Set generator seed before each use
        self.generator = torch.Generator(device=device)
        self.pipe.set_progress_bar_config(disable=True)
    
    


    def forward(self, batch_size, pos_prompt, neg_prompt):
        """Generate images from prompts."""
        batchsize = 2 if batch_size == 5 else 1
        positive_prompts = [pos_prompt] * batchsize
        negative_prompts = [neg_prompt] * batchsize
        
        generated_images = []
        with torch.no_grad():
            for i in range(batchsize):
                batch_output = self.pipe(
                    prompt=positive_prompts[i],
                    negative_prompt=negative_prompts[i],
                    guidance_scale=15,
                    generator=self.generator,
                )
                generated_images.append(batch_output.images[0])
        
        generated_images = torch.stack([ToTensor()(img) for img in generated_images]).to(torch.float16).to(device)
        return generated_images
    
class GenerateUnknownImages(nn.Module):
    """Generate and preprocess unknown images using Stable Diffusion."""
    
    def __init__(self):
        super().__init__()
        self.diffusion = StableDiffusion()
        self.resize_transform, self.normalize = get_image_transforms()

    def forward(self, batch_size, pos_prompt, neg_prompt):
        """Generate normalized unknown images."""
        generated_images = self.diffusion(batch_size, pos_prompt, neg_prompt)
        resized_images = torch.stack([self.resize_transform(x) for x in generated_images])
        normalized_images = self.normalize(resized_images).to(device)
        return normalized_images

    def generate_pil_images(self, batch_size, pos_prompt, neg_prompt):
        """Generate PIL images for saving or visualization."""
        with torch.no_grad():
            generated_images = self.diffusion(batch_size, pos_prompt, neg_prompt)
            pil_images = []
            for img_tensor in generated_images:
                img_tensor = img_tensor.detach().cpu().clamp(0, 1)
                pil_images.append(to_pil_image(img_tensor))
            return pil_images

class CrossAttention(nn.Module):
    """Cross-attention mechanism for feature fusion."""
    # 0.25
    def __init__(self, embed_dim, num_heads, dropout=0.5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, 
            1, 
            batch_first=True, 
            dropout=dropout
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            batch_first=True, 
            dropout=dropout
        )

        self.temperature = nn.Parameter(torch.ones(1))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)

    def forward(self, image_features, attribute_embeddings, mask_embed):
        self_attn_output, _ = self.self_attn(
            attribute_embeddings, 
            attribute_embeddings, 
            attribute_embeddings,
            key_padding_mask=mask_embed
        )
        self_attn_output = self.layer_norm(self_attn_output + attribute_embeddings)
        attn_output, attn_weights = self.multihead_attn(
            image_features, 
            self_attn_output, 
            self_attn_output,
            key_padding_mask=mask_embed
        )
        output = self.layer_norm1(attn_output + image_features)
        
        return self.layer_norm2(self.out_proj(output)+output)


class MLP(nn.Module):
    """Multi-layer perceptron with optional activation and dropout."""
    
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super().__init__()
        self.fc1 = nn.Linear(in_size, mid_size)
        self.fc2 = nn.Linear(mid_size, out_size)
        self.dropout = nn.Dropout(dropout_r) if dropout_r > 0 else None
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self.fc1(x)
        if self.relu:
            x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return self.fc2(x)


class AttFlat(nn.Module):
    """Attention-based feature flattening from MCAN."""
    
    def __init__(self, embed_dim=512):
        super().__init__()
        self.mlp = MLP(embed_dim, embed_dim, 1, dropout_r=0.15, use_relu=True)
        self.linear_merge = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)
        
        att_list = [torch.sum(att[:, :, i:i+1] * x, dim=1) for i in range(1)]
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


class PromptLearner(nn.Module):
    """Learnable prompt generation for CLIP."""
    
    def __init__(self, classnames, clip_model, n_ctx, config):
        super().__init__()
        # seed_everything(42)
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
       
        ctx_vectors = torch.empty(n_ctx - 2, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        
        ctx_vectors_unk = torch.empty(2, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_unk, std=0.02)
        
       
        self.prompt_cls = nn.Sequential(
            nn.Linear(768, config["project_dim"]),
            nn.GELU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["project_dim"], ctx_dim)
        )
        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx_k = nn.Parameter(ctx_vectors_unk)
        
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """Construct prompts by combining context with prefix and suffix."""
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        """Generate learnable prompts."""
        ctx = self.prompt_cls(self.ctx)
        ctx = torch.cat((self.ctx_k, ctx), dim=0)
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        return prompts, self.ctx
class CustomCLIP(nn.Module):
    """Custom CLIP model with prompt learning and style adaptation."""
    
    def __init__(self, classnames: List[str], domainnames: List[str], clip_model: nn.Module, config, gated=False):
        super().__init__()
        # seed_everything(42)
        self.ctx = config["n_ctx"]
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.promptlearner = PromptLearner(classnames, clip_model, self.ctx, config)
        
        
        self.prompt_mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 768)
        )
        
        self.per_class_gate = nn.Parameter(torch.ones(len(classnames) - 1) * 0.5)
        self.cross_attention = CrossAttention(512, config["n_head"])
        self.projector = nn.Linear(512, 512)
        # self.projector = nn.Identity()
        
      
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_class = len(classnames)
        self.classnames = classnames
        self.domainnames = domainnames
        self.gated = gated

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract image features with prompt conditioning."""
        prompts, ctx = self.promptlearner()
        image_features, _, _ = self.image_encoder(image.type(self.dtype), ctx)
        image_features = F.normalize(image_features, dim=-1)
        image_features = self.prompt_mlp(image_features)
        return image_features
    def forward(self, image: torch.Tensor, attri: torch.Tensor, mask_embed: torch.Tensor, 
                label: torch.Tensor = None, dom_label: torch.Tensor = None, batch=None):
        """Forward pass with optional domain adaptation."""
        prompts, ctx = self.promptlearner()
        image_features, _, _ = self.image_encoder(image.type(self.dtype), ctx)
        image_features = F.normalize(image_features, dim=-1)
        
        
        prompt_prefix = " ".join(["X"] * self.ctx)
        tokenized_prompts = torch.cat([
            clip.tokenize(f"{prompt_prefix} {p}") for p in self.classnames
        ]).to(image.device)
        
   
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)
        
 
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        if dom_label is not None:
            txt_features = text_features[:-1, :].repeat(image_features.size(0) - batch, 1)
            
            loss_sty, sty_embedding = self._compute_style_loss(
                image_features[:-batch, :], dom_label[:-batch], label[:-batch], attri, mask_embed, logit_scale
            )
            return logits, loss_sty, txt_features, sty_embedding
        else:
            return logits, image_features
    def _process_domain_features(self, domain_features, cross_atten, domain_img, 
                               tokenized_prompts, logit_scale, original_indices, 
                               sty_embedding_list):
        """Process features for a specific domain."""
        domain_embeddings = []
        domain_logits = []
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        
        for i in range(domain_features.size(0)):
            embedding_copy = embedding.clone()
            
            
            ctx_i = cross_atten[:, i, :].unsqueeze(1)
            
            embedding_copy[:, 1:5, :] += ctx_i
            
            # Compute embeddings and logits
            embedding_int = self.text_encoder(embedding_copy, tokenized_prompts)
            embedding_int = F.normalize(embedding_int, dim=-1)
            logit = logit_scale * domain_features[i] @ embedding_int.t()
            
            domain_logits.append(logit)
            domain_embeddings.append(embedding_int)
        
        # Store results with original indices
        domain_logits = torch.stack(domain_logits)
        domain_embeddings = torch.stack(domain_embeddings)
        
        for i, idx in enumerate(original_indices):
            sty_embedding_list.append((idx.item(), domain_embeddings[i]))
        
        return domain_logits
    @torch.cuda.amp.autocast()
    def _compute_style_loss(self, image_features: torch.Tensor, dom_label: torch.Tensor,
                        label: torch.Tensor, attri: torch.Tensor, mask_embed: torch.Tensor,
                        logit_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute style adaptation loss across domains."""
        device = image_features.device
        sty_embedding_list = []
        
        # Collect all logits and labels from all domains
        all_logits = []
        all_labels = []
    
        for domain in [0, 1, 2]:
            domain_mask = dom_label == domain
            domain_features = image_features[domain_mask]
        
            if domain_features.size(0) == 0:
                continue
            
            original_indices = torch.nonzero(domain_mask, as_tuple=False).squeeze(1)
            domain_labels = label[domain_mask]
        
            domain_name = self.domainnames[domain].replace('_', ' ')
            tokenized_prompts = torch.cat([
                clip.tokenize(f"A {domain_name} of a {p}")
                for p in self.classnames[:-1]
            ]).to(device)
        
            n_cls = len(self.classnames)
            domain_img = einops.repeat(domain_features, 'm n -> k m n', k=n_cls-1)
            cross_atten = self.cross_attention(domain_img, self.projector(attri), mask_embed)
            domain_logits = self._process_domain_features(
                domain_features, cross_atten, domain_img, tokenized_prompts,
                logit_scale, original_indices, sty_embedding_list
            )
            
            # Collect logits and labels instead of computing loss immediately
            all_logits.append(domain_logits)
            all_labels.append(domain_labels)
    
        # Compute single cross-entropy loss if we have any data
        if all_logits:
            # Concatenate all logits and labels
            combined_logits = torch.cat(all_logits, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            
            # Compute single cross-entropy loss
            total_loss = F.cross_entropy(combined_logits, combined_labels)
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
        sty_embedding_list.sort(key=lambda x: x[0])
        sty_embedding = torch.cat([x[1] for x in sty_embedding_list]) if sty_embedding_list else torch.empty(0, device=device)
    
        return total_loss, sty_embedding