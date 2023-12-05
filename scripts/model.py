import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

def create_custom_model(args, model):
    return CustomCLIP(model, args.device, args.use_vit)


class CustomCLIP(nn.Module):
    override = ["clip", "forward", "lock_text_tower", "vit_model", "vit_processor"]

    def __init__(self, clip, device = 'cuda', use_vit = False):
        super().__init__()
        self.clip = clip
        model_ckpt = 'google/vit-base-patch16-224-in21k'
        self.device = device
        self.vit_processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.vit_model = AutoModel.from_pretrained(model_ckpt).to(self.device)
        self.use_vit = use_vit
        self.stored_vit = {}

    def __getattr__(self, name):
        if name in self.override:
            return super().__getattr__(name)
        else:
            return getattr(self.clip, name)

    def lock_text_tower(self, unlocked_layers, freeze_layer_norm):
        # ignore options and just lock the entire text tower
        for param in self.clip.transformer.parameters():
            param.requires_grad = False

    def vit_features(self, images):
        batch_size = len(images)
        vit_images_features = torch.zeros(batch_size, 768, device = self.device) # 768 is the embedding size of ViT
        known_idx = []
        with torch.no_grad():
            images_norm = images - images.min(1, keepdim=True)[0]
            images_norm /= images_norm.max(1, keepdim=True)[0]
            inputs = self.vit_processor(images=images_norm, return_tensors="pt", do_rescale = False).to(self.device)
            vit_images_features = self.vit_model(**inputs).last_hidden_state[:, 0].to(self.device)
        return vit_images_features
    def forward(self, image, text, query=None, keyword=None):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)

        # Whether to use the ViT features to compute cosine similarity when pseudo-labeling
        vit_image_features = self.vit_features(image) if self.use_vit else None

        out = {
            "image_features": image_features,
            "text_features": text_features,
            "vit_image_features": vit_image_features,
            "logit_scale": self.logit_scale.exp()
        }

        if query is not None:  # unlabeled image
            query_features = self.encode_image(query, normalize=True)
            # Like above, whether to use ViT for cosine similarity
            vit_query_features = self.vit_features(image) if self.use_vit else None
            out.update({
                "query_features": query_features,
                "vit_query_features": vit_query_features,
                
            })

        if keyword is not None:  # keyword tokens
            keyword_features = self.encode_text(keyword, normalize=True)
            out.update({
                "keyword_features": keyword_features,
            })

        if self.output_dict:
            return out

        return image_features, text_features, self.logit_scale.exp()