from transformers import PretrainedConfig
import torch

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


@torch.no_grad()
def newton_schulz(G, steps=30):
    G /= torch.linalg.norm(G, dim=(-1, -2), keepdim=True, ord="fro")  # inefficient
    m = G.shape[1]
    I = torch.eye(m, device=G.device).repeat(G.shape[0], 1, 1)
    for _ in range(steps):
        G = torch.bmm((3 * I - torch.bmm(G, G.transpose(-1, -2))), (G / 2))
    return G
