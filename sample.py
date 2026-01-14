import os
from typing import Any
import math

import hydra
import torch
from pytorch_lightning.lite import LightningLite

import numpy as np

import omegaconf
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from torchvision.utils import save_image, make_grid

from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from utils.helpers import ensure_path_join, denormalize_to_zero_to_one

import sys
sys.path.insert(0, 'IDiff-Face/')


class DiffusionSamplerLite(LightningLite):
    def run(self, cfg) -> Any:
        train_cfg_path = os.path.join(cfg.checkpoint.path, '.hydra', 'config.yaml')
        train_cfg = omegaconf.OmegaConf.load(train_cfg_path)

        # do not set seed to get different samples from each device
        self.seed_everything(cfg.sampling.seed * (1 + self.global_rank))

        # instantiate stuff from restoration config
        diffusion_model = instantiate(train_cfg.diffusion)

        # registrate model in lite
        diffusion_model = self.setup(diffusion_model)

        # load state dicts from checkpoint
        if cfg.checkpoint.global_step is not None:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', f'ema_averaged_model_{cfg.checkpoint.global_step}.ckpt')
        if cfg.checkpoint.use_non_ema:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', f'model.ckpt')
        else:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', 'ema_averaged_model.ckpt')

        diffusion_model.module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(checkpoint_path)

        # sample size
        size = (train_cfg.constants.input_channels, train_cfg.constants.image_size, train_cfg.constants.image_size)

        if train_cfg.latent_diffusion:
            # create VQGAN encoder and decoder for training in its latent space
            latent_encoder = VQEncoderInterface(
                first_stage_config_path=os.path.join(".", "neg_prompt", "models", "autoencoder",
                                                     "first_stage_config.yaml"),
                encoder_state_dict_path=os.path.join(".", "neg_prompt", "models", "autoencoder",
                                                     "first_stage_encoder_state_dict.pt")
            )

            size = latent_encoder(torch.ones([1, *size])).shape[-3:]
            del latent_encoder

            latent_decoder = VQDecoderInterface(
                first_stage_config_path=os.path.join(".", "neg_prompt", "models", "autoencoder",
                                                     "first_stage_config.yaml"),
                decoder_state_dict_path=os.path.join(".", "neg_prompt", "models", "autoencoder",
                                                     "first_stage_decoder_state_dict.pt")
            )
            latent_decoder = self.setup(latent_decoder)
            latent_decoder.eval()
        else:
            latent_decoder = None

        contexts_file = ensure_path_join(cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts) + ".npy")
        input_contexts_name = contexts_file.split("/")[-1].split(".")[0]
        model_name = cfg.checkpoint.path.split("/")[-1]

        if cfg.create_contexts.contexts_save_name == "UIFace_":
            contexts = np.load(contexts_file)
            contexts = contexts[: cfg.sampling.n_contexts]

            contexts_norm = np.linalg.norm(contexts, axis=1)
            contexts = contexts / contexts_norm[:, np.newaxis]
            print(f"contexts.shape: {contexts.shape}")

            context_ids = list(i for i in range(0, contexts.shape[0]))        
        else:
            contexts = torch.load(contexts_file)
            contexts = {str(int(k)): v for k, v in contexts.items()}

            assert len(contexts) >= cfg.sampling.n_contexts

            if type(contexts) == dict:
                context_ids = list(contexts.keys())[:cfg.sampling.n_contexts]
            else:
                exit(1)

        if cfg.checkpoint.use_non_ema:
            model_name += "_non_ema"
        elif cfg.checkpoint.global_step is not None:
            model_name += f"_{cfg.checkpoint.global_step}"

        if cfg.neg_prompt.is_adaptive_w:
            if cfg.neg_prompt.is_reverse_adaptive:
                w_path = "reverse_adaptive/w_" + str(10*cfg.neg_prompt.w)
            else: 
                w_path = "adaptive/w_" + str(10*cfg.neg_prompt.w)
        else:
            w_path = "w_" + str(10*cfg.neg_prompt.w)

        if cfg.sampling.is_ddim:
            samples_dir = ensure_path_join(cfg.sampling.root_dir, model_name, input_contexts_name, cfg.sampling.method, w_path, "samples")
        else:
            samples_dir = ensure_path_join(cfg.sampling.root_dir, model_name, input_contexts_name, "ddpm_" + cfg.sampling.method, w_path, "samples")

        length_before_filter = len(context_ids)
        context_ids = list(filter(lambda i: not os.path.isfile(os.path.join(samples_dir, f"{i}.png")), context_ids))
        print(f"Skipped {length_before_filter - len(context_ids)} context ids, because for them files already seem to "
              f"exist!")
        context_ids = self.split_across_devices(context_ids)

        if cfg.neg_prompt.w != 0:
            if cfg.sampling.method == 'cc':
                context_dict = np.load(os.path.join(cfg.sampling.root_dir, cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts),"closest_cossim.npy"), allow_pickle=True).item()
            elif cfg.sampling.method == 'fc':
                context_dict = np.load(os.path.join(cfg.sampling.root_dir, cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts), "farthest_cossim.npy"), allow_pickle=True).item()
            elif cfg.sampling.method == 'mc':
                context_dict = np.load(os.path.join(cfg.sampling.root_dir, cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts), "median_cossim.npy"), allow_pickle=True).item()
            elif cfg.sampling.method == 'ac':
                context_dict = np.load(os.path.join(cfg.sampling.root_dir, cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts), "avg_cossim.npy"), allow_pickle=True).item()

        if self.global_rank == 0:
            with open(ensure_path_join(f"{samples_dir}.yaml"), "w+") as f:
                OmegaConf.save(config=cfg, f=f.name)
            
        for id_name in context_ids:
            prefix = id_name

            context = torch.from_numpy(contexts[id_name])
            context = context.repeat(cfg.sampling.batch_size, 1).cuda()

            if cfg.neg_prompt.w != 0:
                if cfg.sampling.method == 'rand':
                    if id_name=='0':
                        negative_context = torch.from_numpy(contexts[str(len(contexts)-1)])
                    else:
                        negative_context = torch.from_numpy(contexts[str(int(id_name)-1)])
                elif cfg.sampling.method == 'empty':
                    negative_context = torch.zeros_like(torch.from_numpy(contexts['0']))
                else:
                    closest_id_name = context_dict[int(prefix)].item()
                    negative_context = torch.from_numpy(contexts[str(closest_id_name)])

                negative_context = negative_context.repeat(cfg.sampling.batch_size, 1).cuda()

                self.perform_sampling(
                    diffusion_model=diffusion_model,
                    n_samples=cfg.sampling.n_samples_per_context,
                    size=size,
                    batch_size=cfg.sampling.batch_size,
                    samples_dir=samples_dir,
                    prefix=prefix,
                    context=context,
                    negative_context=negative_context, 
                    w=cfg.neg_prompt.w,
                    is_adaptive_w=cfg.neg_prompt.is_adaptive_w,
                    is_reverse_adaptive=cfg.neg_prompt.is_reverse_adaptive,
                    ddim_step=cfg.sampling.ddim_step,
                    is_ddim=cfg.sampling.is_ddim,
                    latent_decoder=latent_decoder
                )
            else:
                self.perform_original_sampling(
                    diffusion_model=diffusion_model,
                    n_samples=cfg.sampling.n_samples_per_context,
                    size=size,
                    batch_size=cfg.sampling.batch_size,
                    samples_dir=samples_dir,
                    prefix=prefix,
                    context=context,
                    ddim_step=cfg.sampling.ddim_step,
                    is_ddim=cfg.sampling.is_ddim,
                    latent_decoder=latent_decoder
                )

    @staticmethod
    def perform_original_sampling(
            diffusion_model, n_samples, size, batch_size, samples_dir,
            prefix: str = None, context: torch.Tensor = None, ddim_step: int = 200, is_ddim: bool = True,
            latent_decoder: torch.nn.Module = None):

        n_batches = math.ceil(n_samples / batch_size)

        samples_for_grid = []

        if context is not None:
            assert prefix is not None

        with torch.no_grad():
            for _ in range(n_batches):

                if is_ddim:
                    batch_samples = diffusion_model.original_sample_ddim(batch_size, size, context=context, ddim_step=ddim_step)
                else:
                    batch_samples = diffusion_model.original_sample_ddpm(batch_size, size, context=context)

                with torch.no_grad():
                    if latent_decoder:
                        batch_samples = latent_decoder(batch_samples).cpu()

                batch_samples = denormalize_to_zero_to_one(batch_samples)

                samples_for_grid.append(batch_samples)

            samples = torch.cat(samples_for_grid, dim=0)[:n_samples]
            grid = make_grid(samples, nrow=4, padding=0)
            save_image(grid, ensure_path_join(samples_dir, f"{prefix}.png"))

    @staticmethod
    def perform_sampling(
            diffusion_model, n_samples, size, batch_size, samples_dir,
            prefix: str = None, context: torch.Tensor = None, 
            negative_context: torch.Tensor = None, w: float = 0.5, is_adaptive_w: bool = False, is_reverse_adaptive: bool = False, ddim_step: int = 200, is_ddim: bool = True,
            latent_decoder: torch.nn.Module = None):

        n_batches = math.ceil(n_samples / batch_size)

        samples_for_grid = []

        if context is not None:
            assert prefix is not None

        with torch.no_grad():
            for _ in range(n_batches):

                if is_ddim:
                    batch_samples = diffusion_model.sample_ddim(batch_size, size, context=context, negative_context=negative_context, w=w, is_adaptive_w=is_adaptive_w, is_reverse_adaptive=is_reverse_adaptive, ddim_step=ddim_step)
                else:
                    batch_samples = diffusion_model.sample_ddpm(batch_size, size, context=context, negative_context=negative_context, w=w, is_adaptive_w=is_adaptive_w, is_reverse_adaptive=is_reverse_adaptive)

                with torch.no_grad():
                    if latent_decoder:
                        batch_samples = latent_decoder(batch_samples).cpu()

                batch_samples = denormalize_to_zero_to_one(batch_samples)

                samples_for_grid.append(batch_samples)

            samples = torch.cat(samples_for_grid, dim=0)[:n_samples]
            grid = make_grid(samples, nrow=4, padding=0)
            save_image(grid, ensure_path_join(samples_dir, f"{prefix}.png"))

    def split_across_devices(self, L):
        if type(L) is int:
            L = list(range(L))

        chunk_size = math.ceil(len(L) / self.world_size)
        L_per_device = [L[idx: idx + chunk_size] for idx in range(0, len(L), chunk_size)]
        while len(L_per_device) < self.world_size:
            L_per_device.append([])

        return L_per_device[self.global_rank]

    @staticmethod
    def spherical_interpolation(value, start, target):
        start = torch.nn.functional.normalize(start)
        target = torch.nn.functional.normalize(target)
        omega = torch.acos((start * target).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0 - value) * omega) / so).unsqueeze(1) * start + (
                torch.sin(value * omega) / so).unsqueeze(1) * target
        return res


@hydra.main(config_path='configs', config_name='sample_config', version_base=None)
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = DiffusionSamplerLite(devices="auto", accelerator="auto")
    sampler.run(cfg)


if __name__ == "__main__":

    sample()
