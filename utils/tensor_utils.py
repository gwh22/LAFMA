import torch
import torch.distributed as dist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
import json
import models.tta.hifigan as hifigan


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def tensors_to_np(tensors):
    if isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = tensors_to_np(v)
        new_np = v
    else:
        raise Exception(f"tensors_to_np does not support type {type(tensors)}.")
    return new_np


def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret


def move_to_cuda(batch, gpu_id=0):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, "cuda", None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif callable(getattr(batch, "to", None)):
        return batch.to(torch.device("cuda", gpu_id), non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    return batch


def log_metrics(logger, metrics, step=None):
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        logger.add_scalar(k, v, step)


def spec_to_figure(spec, vmin=None, vmax=None, title="", f0s=None, dur_info=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    H = spec.shape[1] // 2
    fig = plt.figure(figsize=(12, 6), dpi=100)
    plt.title(title)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    if dur_info is not None:
        assert isinstance(dur_info, dict)
        txt = dur_info["txt"]
        dur_gt = dur_info["dur_gt"]
        if isinstance(dur_gt, torch.Tensor):
            dur_gt = dur_gt.cpu().numpy()
        dur_gt = np.cumsum(dur_gt).astype(int)
        for i in range(len(dur_gt)):
            shift = (i % 8) + 1
            plt.text(dur_gt[i], shift * 4, txt[i])
            plt.vlines(dur_gt[i], 0, H // 2, colors="b")  # blue is gt
        plt.xlim(0, dur_gt[-1])
        if "dur_pred" in dur_info:
            dur_pred = dur_info["dur_pred"]
            if isinstance(dur_pred, torch.Tensor):
                dur_pred = dur_pred.cpu().numpy()
            dur_pred = np.cumsum(dur_pred).astype(int)
            for i in range(len(dur_pred)):
                shift = (i % 8) + 1
                plt.text(dur_pred[i], H + shift * 4, txt[i])
                plt.vlines(dur_pred[i], H, H * 1.5, colors="r")  # red is pred
            plt.xlim(0, max(dur_gt[-1], dur_pred[-1]))
    # if f0s is not None:
    #     ax = plt.gca()
    #     ax2 = ax.twinx()
    #     if not isinstance(f0s, dict):
    #         f0s = {"f0": f0s}
    #     for i, (k, f0) in enumerate(f0s.items()):
    #         if isinstance(f0, torch.Tensor):
    #             f0 = f0.cpu().numpy()
    #         ax2.plot(f0, label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.5)
    #     ax2.set_ylim(0, 1000)
    #     ax2.legend()
    return fig


def get_vocoder(config, device):
    ROOT = "/work/gwh/Amphion/ckpts/tta/hifigan"

    model_path = os.path.join(ROOT, "hifigan_16k_64bins")
    with open(model_path + ".json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)

    ckpt = torch.load(model_path + ".ckpt")
    ckpt = torch_version_orig_mod_remove(ckpt)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def torch_version_orig_mod_remove(state_dict):
    new_state_dict = {}
    new_state_dict["generator"] = {}
    for key in state_dict["generator"].keys():
        if "_orig_mod." in key:
            new_state_dict["generator"][key.replace("_orig_mod.", "")] = state_dict[
                "generator"
            ][key]
        else:
            new_state_dict["generator"][key] = state_dict["generator"][key]
    return new_state_dict


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    return wavs
