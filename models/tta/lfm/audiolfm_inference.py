# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import lightning as L
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from utils.utils import num_parameters
from lightning.fabric import Fabric

from models.tta.autoencoder.autoencoder import AutoencoderKL
from models.tta.lfm.audioldm import AudioLDM

from transformers import T5EncoderModel, AutoTokenizer
from models.tta.lfm.fm_scheduler import (
    FlowMatchingTrainer,
)
from utils.io import save_audio
import datetime

# from diffusers import DDPMScheduler, PNDMScheduler, DDIMScheduler
from utils.tensor_utils import (
    get_vocoder,
    vocoder_infer,
)


class LAFMA_Inference(object):
    def __init__(self, config, loggers=None, precision="32-true") -> None:
        self.fabric = Fabric(
            accelerator=config.train.accelerator,
            strategy=config.train.strategy,
            devices=config.train.devices,
            loggers=loggers,
            precision=precision,
        )
        self.cfg = config

        self.fabric.launch()
        self.vocoder = get_vocoder(None, "cpu")

    def _build_model(self):
        if self.cfg.infer:
            with self.fabric.init_module(empty_init=True):
                # audioldm
                model = AudioLDM(self.cfg.model.audioldm)
                # pretrained autoencoder
                autoencoder = AutoencoderKL(self.cfg.model.autoencoderkl)
                autoencoder_path = self.cfg.model.autoencoder_path
                checkpoint = self.fabric.load(autoencoder_path)
                autoencoder.load_state_dict(checkpoint["autoencoder"])
                autoencoder.requires_grad_(requires_grad=False)
                autoencoder.eval()
                # pretrained text encoder
                tokenizer = AutoTokenizer.from_pretrained(
                    "huggingface/flan-t5-large",
                    model_max_length=512,
                )
                text_encoder = T5EncoderModel.from_pretrained(
                    "huggingface/flan-t5-large",
                )
                text_encoder.requires_grad_(requires_grad=False)
                text_encoder.eval()

        if self.fabric.local_rank == 0:
            output_dir = self.cfg.test_out_dir
            model_file = os.path.join(output_dir, "model.log")
            if os.path.exists(model_file):
                os.remove(model_file)
            log = open(model_file, mode="a+", encoding="utf-8")
            print(model, file=log)
            log.close()
        self.fabric.barrier()
        return model, autoencoder, text_encoder, tokenizer

    @torch.no_grad()
    def mel_to_latent(self, melspec):
        posterior = self.autoencoder.encode(melspec)
        latent = posterior.sample()  # (B, 4, 5, 78)
        return latent

    @torch.no_grad()
    def get_text_embedding(self, text_input_ids, text_attention_mask):
        text_embedding = self.text_encoder(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        )[0]
        return text_embedding  # (B, T, 768)

    def before_test(self):
        self.fabric.barrier()
        test = f"test_out-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.cfg.test_out_dir = os.path.join(self.cfg.train.out_dir, test)
        os.makedirs(self.cfg.test_out_dir, exist_ok=True)

    def _prepare_model(self):
        checkpoint_file = self.cfg.checkpoint_file
        self.fabric.print(f"start test from {checkpoint_file}.")
        if checkpoint_file.endswith(".ckpt"):
            checkpoint = self.fabric.load(checkpoint_file)
            self.model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("supported checkpoint file format : .ckpt")

        output_dir = self.cfg.test_out_dir

        if self.fabric.local_rank == 0:
            if self.cfg.infer:
                log_file = os.path.join(output_dir, "run.log")
            else:
                log_file = os.path.join(output_dir, "logs/run.log")
            log = open(log_file, mode="a+", encoding="utf-8")
            print("----------------------", file=log)
            print(f"accelerator: {self.fabric.accelerator}", file=log)
            print(f"strategy: {self.fabric.strategy}", file=log)
            print("----------------------", file=log)
            print(f"use dataset: {self.cfg.dataset}", file=log)
            print(f"sampling rate: {self.cfg.preprocess.sample_rate}", file=log)
            print(f"project name: {self.cfg.train.project}", file=log)
            print("----------------------", file=log)
            log.close()
        self.fabric.print("----------------------")
        self.fabric.print(f"accelerator: {self.fabric.accelerator}")
        self.fabric.print(f"strategy: {self.fabric.strategy}")
        self.fabric.print("----------------------")
        self.fabric.print(f"use dataset: {self.cfg.dataset}")
        self.fabric.print(f"sampling rate: {self.cfg.preprocess.sample_rate}")
        self.fabric.print(f"project name: {self.cfg.train.project}")
        self.fabric.barrier()

    def test(self):
        self.before_test()
        self.model, self.autoencoder, self.text_encoder, self.tokenizer = (
            self._build_model()
        )
        self.model = self.fabric.setup(self.model)
        self.trainer = FlowMatchingTrainer(self.model, sample_N=self.cfg.num_steps)

        self._prepare_model()

        # test
        self.test_step()
        self.fabric.print("-" * 16)
        if self.fabric.device.type == "cuda":
            self.fabric.print(
                f"memory used: {torch.cuda.max_memory_allocated()/1e9:.02f} GB"
            )

    def test_step(self):
        assert self.vocoder is not None, "Vocoder is not loaded."
        if self.cfg.infer_text is not None:
            out_dir = self.cfg.test_out_dir
            os.makedirs(out_dir, exist_ok=True)

            pred_audio = self.inference_for_single_utterance()
            save_path = os.path.join(out_dir, "test_pred.wav")
            save_audio(save_path, pred_audio, self.cfg.preprocess.sample_rate)

    @torch.inference_mode()
    def inference_for_single_utterance(self):
        text = self.cfg.infer_text

        text_input = self.tokenizer(
            [text],
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            padding="do_not_pad",
            return_tensors="pt",
        )

        text_input = self.fabric.to_device(text_input)
        text_embedding = self.text_encoder(text_input.input_ids)[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * 1,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_input = self.fabric.to_device(uncond_input)
        uncond_embedding = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embedding, text_embedding])

        guidance_scale = self.cfg.guidance_scale

        self.model.eval()

        # sample
        latents_t = torch.randn(
            (
                1,
                8,
                256,
                16,
            )
        )
        latents_out, nfe = self.trainer.euler_sample(
            text_embeddings, latents_t.shape, guidance_scale
        )

        print(latents_out.shape, nfe)

        with torch.no_grad():
            mel_pred = self.autoencoder.decode(latents_out)
        print(mel_pred.shape)
        wav_pred = vocoder_infer(mel_pred.transpose(2, 3)[0].cpu(), self.vocoder)
        wav_pred = (
            wav_pred / np.max(np.abs(wav_pred))
        ) * 0.8  # Normalize the energy of the generation output
        return wav_pred
