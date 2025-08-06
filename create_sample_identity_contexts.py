import torch
from utils.helpers import ensure_path_join
import numpy as np

from typing import Any
import hydra
from pytorch_lightning.lite import LightningLite
from omegaconf import OmegaConf, DictConfig

class CreateContextsLite(LightningLite):
    def run(self, cfg) -> Any:
        n_contexts = cfg.create_contexts.n_contexts

        random_uniform_embeddings = sample_synthetic_uniform_embeddings(n_contexts)
        torch.save(random_uniform_embeddings, ensure_path_join(cfg.create_contexts.contexts_save_path, f"random_synthetic_uniform_{n_contexts}.npy"))
        del random_uniform_embeddings

def sample_synthetic_uniform_embeddings(n_contexts):
    embeddings = torch.nn.functional.normalize(torch.randn([n_contexts, 512])).numpy()
    return {str(id_name): id_embedding for id_name, id_embedding in enumerate(embeddings)} 

@hydra.main(config_path='configs', config_name='sample_config', version_base=None)
def create_contexts(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = CreateContextsLite(devices="auto", accelerator="auto")
    sampler.run(cfg)

if __name__ == "__main__":
    create_contexts()