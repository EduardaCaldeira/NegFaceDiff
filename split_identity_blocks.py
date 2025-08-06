import os
from typing import Any

import torch
import torchvision
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

import hydra
from pytorch_lightning.lite import LightningLite
from omegaconf import OmegaConf, DictConfig
from utils.helpers import ensure_path_join

from PIL import Image

import numpy as np

import sys

def split_identity_grid(samples_dir, id_name, image_size: int):

    with open(os.path.join(samples_dir, id_name), "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

    img = torchvision.transforms.functional.to_tensor(img)

    nrows = img.shape[1] // image_size
    ncols = img.shape[2] // image_size

    id_images = []
    for r in range(nrows):
        for c in range(ncols):
            tile = img[:, r * image_size: (r + 1) * image_size, c * image_size: (c + 1) * image_size]
            id_images.append(tile)

    return id_images

class SplittingBlocks(LightningLite):
    def run(self, cfg) -> Any:
        # initializing variables  
        n_contexts = cfg.create_contexts.n_contexts
        contexts_file = ensure_path_join(cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts) + ".npy")
        contexts_name = contexts_file.split("/")[-1].split(".")[0]
        model_name = cfg.checkpoint.path.split("/")[-1]

        w_path = "w_" + str(10*cfg.neg_prompt.w)

        if cfg.sampling.is_ddim:
            root_dir = ensure_path_join(cfg.sampling.root_dir, model_name, contexts_name, cfg.sampling.method, w_path)
        else:
            root_dir = ensure_path_join(cfg.sampling.root_dir, model_name, contexts_name, "ddpm_" + cfg.sampling.method, w_path)

        blocks_dir = ensure_path_join(root_dir, "samples")
        dataset_dir = ensure_path_join(root_dir, "dataset/original")
        
        for id_block_file in os.listdir(blocks_dir):
            if not id_block_file.lower().endswith(".png"):  # only process PNG files
                print(f"Skipping non-PNG file: {id_block_file}")
                continue
            if os.path.isfile(os.path.join(blocks_dir, id_block_file)): # only files are evaluated
                id_images = split_identity_grid(blocks_dir, id_block_file, image_size=128)
                id_name = id_block_file.split(".")[0]

                for i, img in enumerate(id_images):
                    os.makedirs(dataset_dir, exist_ok=True)
                    save_image(img, ensure_path_join(dataset_dir, f"{id_name}/{i}.png"))

@hydra.main(config_path='configs', config_name='sample_config', version_base=None)
def split(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = SplittingBlocks(devices="auto", accelerator="auto")
    sampler.run(cfg)

if __name__ == "__main__":
    split()