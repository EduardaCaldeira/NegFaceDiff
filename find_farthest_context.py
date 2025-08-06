import os
from typing import Any
import math

import hydra
import torch
from pytorch_lightning.lite import LightningLite

import torch.nn.functional as F
import numpy as np
import re
import matplotlib.pyplot as plt

import omegaconf
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from torchvision.utils import save_image, make_grid

from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from utils.helpers import ensure_path, ensure_path_join, denormalize_to_zero_to_one

import sys
sys.path.insert(0, 'IDiff-Face/')

class CenterGeneratorLite(LightningLite):
    def run(self, cfg) -> Any:
        contexts_file = ensure_path_join(cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts) + ".npy")
        contexts = torch.load(contexts_file)
        context_ids = list(contexts.keys())[:cfg.sampling.n_contexts]

        contexts_matrix = torch.from_numpy(np.stack([contexts[key] for key in context_ids]))

        input_contexts_name = contexts_file.split("/")[-1].split(".")[0]
        model_name = cfg.checkpoint.path.split("/")[-1]

        save_dir = os.path.join(cfg.sampling.root_dir, cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts))
        os.makedirs(save_dir, exist_ok=True)

        dict_closest_cossim = {}

        # determining the farthest imposter context for each context
        for prefix in range(0, contexts_matrix.shape[0]):
            current_context = contexts_matrix[prefix, :]

            # computing the cosine similarity between the current context and all other contexts
            cos_sim = F.cosine_similarity(current_context.unsqueeze(0), contexts_matrix, dim=1)
            closest_context_idx = torch.argmin(cos_sim)

            dict_closest_cossim[prefix] = closest_context_idx

        np.save(os.path.join(save_dir, "farthest_cossim.npy"), dict_closest_cossim)

@hydra.main(config_path='configs', config_name='sample_config', version_base=None)
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = CenterGeneratorLite(devices="auto", accelerator="auto")
    sampler.run(cfg)

if __name__ == "__main__":

    sample()
