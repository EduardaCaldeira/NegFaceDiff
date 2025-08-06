import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from os.path import join as ojoin
from torch.utils.data import Dataset, DataLoader

import hydra
from pytorch_lightning.lite import LightningLite
from omegaconf import OmegaConf, DictConfig
from utils.helpers import ensure_path_join
from typing import Any

from utils.align_trans import norm_crop

from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    select_largest=True, min_face_size=60, post_process=False, device="cuda:0"
)

def load_syn_paths(datadir, num_imgs=0):
    img_files = sorted(os.listdir(datadir))
    img_files = img_files if num_imgs == 0 else img_files[:num_imgs]
    return [ojoin(datadir, f_name) for f_name in img_files]


def load_real_paths(datadir, num_imgs=0):
    img_paths = []
    id_folders = sorted(os.listdir(datadir))
    for id in id_folders:
        img_files = sorted(os.listdir(ojoin(datadir, id)))
        img_paths += [ojoin(datadir, id, f_name) for f_name in img_files]
    img_paths = img_paths if num_imgs == 0 else img_paths[:num_imgs]
    return img_paths


def is_folder_structure(datadir):
    """checks if datadir contains folders (like CASIA) or images (synthetic datasets)"""
    img_path = sorted(os.listdir(datadir))[0]
    img_path = ojoin(datadir, img_path)
    return os.path.isdir(img_path)


class InferenceDataset(Dataset):
    def __init__(self, datadir, num_imgs=0, folder_structure=False):
        """Initializes image paths"""
        self.folder_structure = folder_structure
        if self.folder_structure:
            self.img_paths = load_real_paths(datadir, num_imgs)
        else:
            self.img_paths = load_syn_paths(datadir, num_imgs)

        self.img_paths = [path for path in self.img_paths if self.is_valid_image(path)]
        print("Amount of images:", len(self.img_paths))

    def __getitem__(self, index):
        """Reads an image from a file and corresponding label and returns."""
        img_path = self.img_paths[index]
        img_file = os.path.split(img_path)[-1]
        if self.folder_structure:
            tmp = os.path.dirname(img_path)
            img_file = ojoin(os.path.basename(tmp), img_file)

        img = cv2.imread(self.img_paths[index])
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            return None, None 

        return img, img_file

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)

    def is_valid_image(self, file_path):
        return os.path.splitext(file_path)[-1].lower()==".png"

def align_images(folder, batchsize, num_imgs=0, evalDB=False):
    """MTCNN alignment for all images in complete_in_folder and save to complete_out_folder
    args:
            folder: folder path with directories for the original images and where the aligned images will be saved
            batchsize: batch size
            num_imgs: amount of images to align - 0: align all images
            evalDB: evaluation DB alignment
    """

    complete_in_folder = ensure_path_join(folder, "original")
    complete_out_folder = ensure_path_join(folder, "aligned")

    for id_folder in os.listdir(complete_in_folder):
        in_dir = os.path.join(complete_in_folder, id_folder)
        out_dir = ensure_path_join(complete_out_folder, id_folder)

        os.makedirs(out_dir, exist_ok=True)
        is_folder = is_folder_structure(in_dir)
        train_dataset = InferenceDataset(
            in_dir, num_imgs=num_imgs, folder_structure=is_folder
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batchsize, shuffle=False, drop_last=False
        )
        skipped_imgs = []

        for img_batch, img_names in tqdm(train_loader):
            img_batch = img_batch.to("cuda:0")
            boxes, probs, landmarks = mtcnn.detect(img_batch, landmarks=True)

            img_batch = img_batch.detach().cpu().numpy()

            for img, img_name, landmark in zip(img_batch, img_names, landmarks):
                if img is None or landmark is None or not img_name.lower().endswith(".png"):
                    skipped_imgs.append(img_name)
                    continue

                out_path = out_dir
                if is_folder:
                    id_dir = os.path.split(img_name)[0]
                    out_path = ojoin(out_dir, id_dir)
                    os.makedirs(out_path, exist_ok=True)
                    img_name = os.path.split(img_name)[1]

                facial5points = np.array(landmark[0], dtype=np.float64)
                warped_face = norm_crop(
                    img, landmark=facial5points, image_size=112, createEvalDB=evalDB
                )
                cv2.imwrite(os.path.join(out_path, img_name), warped_face)
        print(skipped_imgs)
        print(f"Images with no Face: {len(skipped_imgs)}")

class AlignLite(LightningLite):
    def run(self, cfg) -> Any:
        # initializing variables
        model_name = cfg.checkpoint.path.split("/")[-1]
        contexts_file = ensure_path_join(cfg.create_contexts.contexts_save_path, cfg.create_contexts.contexts_save_name + str(cfg.create_contexts.n_contexts) + ".npy")
        input_contexts_name = contexts_file.split("/")[-1].split(".")[0]

        w_path = "w_" + str(10*cfg.neg_prompt.w)

        if cfg.sampling.is_ddim:
            root_dir = ensure_path_join(cfg.sampling.root_dir, model_name, input_contexts_name, cfg.sampling.method, w_path)
        else:
            root_dir = ensure_path_join(cfg.sampling.root_dir, model_name, input_contexts_name, "ddpm_" + cfg.sampling.method, w_path)

        # initializing arguments
        parser = argparse.ArgumentParser(description="MTCNN alignment")
        parser.add_argument(
        "--folder",
        type=str,
        default=ensure_path_join(root_dir, "dataset"),
        help="folder with images",
        )

        parser.add_argument("--batchsize", type=int, default=32)

        parser.add_argument("--evalDB", type=int, default=0, help="1 for eval DB alignment")

        parser.add_argument(
            "--num_imgs",
            type=int,
            default=0,
            help="amount of images to align; 0 for all images",
        )

        args = parser.parse_args()
        align_images(
            args.folder,
            args.batchsize,
            num_imgs=args.num_imgs,
            evalDB=args.evalDB == 1,
        )

@hydra.main(config_path='configs', config_name='sample_config', version_base=None)
def align(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = AlignLite(devices="auto", accelerator="auto")
    sampler.run(cfg)

if __name__ == "__main__":
    align()