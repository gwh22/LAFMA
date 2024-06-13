# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from models.tta.lfm.audiolfm_inference import LAFMA_Inference
from utils.util import load_config


def build_inference(cfg):
    supported_inference = {"AudioLFM": LAFMA_Inference}

    inference = supported_inference[cfg.train.project](
        cfg, precision=cfg.train.precision
    )

    return inference


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )

    parser.add_argument(
        "--devices", nargs="+", type=int, default=None, help="gpu devices."
    )

    parser.add_argument("--batch_size", type=int, default=None, help="batch size.")

    parser.add_argument(
        "--infer", action="store_true", default=False, help="test mode."
    )

    parser.add_argument(
        "--text",
        help="Text to be synthesized",
        type=str,
        default="",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="final_checkpoint.ckpt",
        help="Checkpoint for test.(only test)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="The total number of denosing steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="The scale of classifer free guidance",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.infer:
        config.checkpoint_file = args.checkpoint_file
    config.infer = args.infer
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.guidance_scale is not None:
        config.guidance_scale = args.guidance_scale
    if args.text is not None:
        config.infer_text = args.text
    if args.devices is not None:
        config.train.devices = args.devices

    if config.infer:
        inferencer = build_inference(config)
        inferencer.test()


if __name__ == "__main__":
    main()
