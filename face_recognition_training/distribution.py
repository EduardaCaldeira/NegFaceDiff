#!/usr/bin/env python
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import re
import math
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.nn.parallel.distributed import DistributedDataParallel

from utils.dataset import DataLoaderX
from backbones.iresnet import iresnet100
import config.config as cfg

from limited_image_folder_dataset import LimitedWidthAndDepthImageFolder

def main(args):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True

    if args.model != "None":
        cfg.model = args.model 

    if args.embedding_type != "None":
        cfg.embedding_type = args.embedding_type

    cfg.width = 1000
    cfg.depth = 20

    seed_path = "w_"+str(cfg.w * 10)

    if cfg.is_ddim:
        root_folder = os.path.join(cfg.root_dir, cfg.model, cfg.embedding_type, cfg.method, seed_path)
    else:
        root_folder = os.path.join(cfg.root_dir, cfg.model, cfg.embedding_type, "ddpm_" + cfg.method, seed_path)

    dataset_folder = os.path.join(root_folder, "dataset/aligned")

    rec = os.path.join(dataset_folder)  # training dataset
    output_dir = os.path.join(root_folder, "FR", "distribution")
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )
    
    trainset = LimitedWidthAndDepthImageFolder(rec, transform=transform, width=cfg.width, depth=cfg.depth)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model_class = iresnet100
    model = model_class(num_features=cfg.embedding_size, dropout=cfg.dropout_ratio).to(local_rank)
    
    try:
        backbone_pth = os.path.join(cfg.root_dir, "ElasticCos.pth")
        model.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        exit(1)

    for ps in model.parameters():
        dist.broadcast(ps, 0)

    model = DistributedDataParallel(module=model, broadcast_buffers=False, device_ids=[local_rank])
    model.eval()

    # evaluation on a pre-trained FR network
    label_features_dict = {}
    train_sampler.set_epoch(0)
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            features = F.normalize(model(images))

            for label, feature in zip(labels.tolist(), features):
                feature = feature.unsqueeze(0)

                if label in label_features_dict:
                    label_features_dict[label].append(feature)
                else:
                    label_features_dict[label] = [feature]

    # gather all labels from the used GPUs
    local_labels = set(label_features_dict.keys()) 
    gathered_labels = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_labels, local_labels)

    # merge all gathered labels into a single set on one GPU
    unique_labels = set()
    for labels in gathered_labels:
        unique_labels.update(labels)

    # process each label individually to avoid memory overflow
    for label in unique_labels:
        local_label_data = [(label, feature) for feature in label_features_dict.get(label, [])]
        gathered_label_data = [None for _ in range(dist.get_world_size())]
        
        # gather only the current label’s features
        dist.all_gather_object(gathered_label_data, local_label_data)

        # merge the gathered features accross all GPUs
        label_features = []
        for gpu_data in gathered_label_data:
            for lbl, feature in gpu_data:
                label_features.append(feature.cuda(local_rank, non_blocking=True))

        merged_features = torch.cat(label_features, dim=0)

        # save the label and its features and free memory
        torch.save(merged_features.detach(), os.path.join(output_dir, f"{label}.pt"))
        del label_features, merged_features
        torch.cuda.empty_cache()

    genuine_scores = determine_genuine_scores(output_dir, output_dir)
    imposter_scores = determine_imposter_scores(output_dir, output_dir)
    plot_scores(genuine_scores, imposter_scores, output_dir)

def plot_scores(genuine_scores, imposter_scores, out_path):
    bins = np.linspace(-1, 1, 100)

    # Compute histogram counts
    gen_counts, _ = np.histogram(genuine_scores, bins=bins)
    imp_counts, _ = np.histogram(imposter_scores, bins=bins)

    # Convert counts to percentage
    gen_counts = (gen_counts / gen_counts.sum()) * 100
    imp_counts = (imp_counts / imp_counts.sum()) * 100

    # Plot histograms with percentage normalization
    plt.hist(bins[:-1], bins=bins, weights=gen_counts, alpha=0.6, color='blue', label="Genuine Pairs")
    plt.hist(bins[:-1], bins=bins, weights=imp_counts, alpha=0.6, color='red', label="Imposter Pairs")

    # Format y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Labels and title
    plt.xlabel("Similarity Score")
    plt.ylabel("Percentage of Instances")
    plt.title("Histogram of Genuine and Imposter Pairs Similarity Scores")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
    plt.savefig(os.path.join(out_path, "similarity.png"))

def determine_genuine_scores(emb_path, out_path):
    cos_sim_gen = []

    files = os.listdir(emb_path)
    pattern = re.compile(r"(\d+)\.pt")
    files = sorted([f for f in files if pattern.match(f)], key=lambda x: int(pattern.match(x).group(1)))

    for file in files:
        # extracting and flattening the embeddings
        embeddings = torch.load(os.path.join(emb_path, file))  
        embeddings = embeddings.view(embeddings.shape[0], -1) 

        # computing pairwise cosine similarity using matrix multiplication (normalization + multiplication)
        norm_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True) 
        cos_sim_matrix = torch.mm(norm_embeddings, norm_embeddings.T) 

        # extracting upper triangular values (excluding diagonal) to avoid duplicates and self-comparisons
        indices = torch.triu_indices(embeddings.size(0), embeddings.size(0), offset=1)
        cos_sim_values = cos_sim_matrix[indices[0], indices[1]].cpu().detach().numpy()
        
        cos_sim_gen.extend(cos_sim_values)

    np.save(os.path.join(out_path, "cos_sim_gen"), np.array(cos_sim_gen))
    np.savetxt(os.path.join(out_path, "cos_sim_gen.txt"), np.array(cos_sim_gen), fmt="%.4f")
    return np.array(cos_sim_gen)
            
def determine_imposter_scores(emb_path, out_path):
    cos_sim_imp = []

    files = os.listdir(emb_path)
    pattern = re.compile(r"(\d+)\.pt")
    files = sorted([f for f in files if pattern.match(f)], key=lambda x: int(pattern.match(x).group(1)))  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # iterating through the contexts in the outer loop to get the first context
    for file_idx, file in enumerate(files):
        embeddings = torch.load(os.path.join(emb_path, file), map_location=device)  # extracting the embedding vector for the first context
        embeddings = embeddings.view(embeddings.shape[0], -1) 

        # computing pairwise cosine similarity using matrix multiplication (normalization + multiplication)
        norm_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        total_files_for_comparison = len(files) - file_idx - 1
        if math.floor(0.05 * total_files_for_comparison) == 0:
            continue
        
        # only ~5% of the files are randomly selected for comparison
        pruned_files_for_comparison = np.random.randint(file_idx + 1, len(files), math.floor(0.05 * total_files_for_comparison))

        # iterating through the contexts in the inner loop to get the second context
        for second_file_idx, second_file in enumerate(files):
            if second_file_idx in pruned_files_for_comparison.tolist():
                second_embeddings = torch.load(os.path.join(emb_path, second_file), map_location=device)
                second_embeddings = second_embeddings.view(second_embeddings.shape[0], -1) 
                second_norm_embeddings = second_embeddings / second_embeddings.norm(dim=1, keepdim=True)

                cos_sim_matrix = torch.mm(norm_embeddings, second_norm_embeddings.T).flatten()

                cos_sim_imp.extend(cos_sim_matrix.cpu().numpy())
    
    max_len = len(cos_sim_imp)
    cos_sim_imp = cos_sim_imp[:int(max_len)]
    
    np.save(os.path.join(out_path, "cos_sim_imp"), np.array(cos_sim_imp))
    np.savetxt(os.path.join(out_path, "cos_sim_imp.txt"), np.array(cos_sim_imp), fmt="%.4f")
    return cos_sim_imp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")

    parser.add_argument("--model", type=str, default="None", help="name of generative difusion model, e.g. unet-cond-ca-bs512-150K")
    parser.add_argument("--embedding_type", type=str, default="None", help="name of embedding type")

    args = parser.parse_args()
    main(args)