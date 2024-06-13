# LAFMA
Official implementation of the paper "LAFMA: A Latent Flow Matching Model for Text-to-Audio Generation" (INTERSPEECH 2024).  [Paper Link](https://arxiv.org) and [Demo Page](https://lafma.github.io) .

## Checkpoints

[VAEGAN Model](https://drive.google.com/file/d/1FRTMxcKHafTcDvEK-c_zRYckF25-UbjQ/view?usp=drive_link): 
The VAEGAN model is the audio VAE that compresses the audio mel-spectrogram into an audio latent.

[LAFMA Model](https://drive.google.com/file/d/1lpX8rN1GvDar4quoLfofI0UireVmuHay/view?usp=drive_link): 
The LAFAM model is the latent flow matching model for text guided audio generation model.

We use the checkpoint of HiFi-GAN vocoder provided by [AudioLDM](https://zenodo.org/records/7884686) .

## Inference
```
# install dependicies
pip install -r requirement.txt

# infer
(first download the huggingface flan-t5-large to the huggingface/flan-t5-large dir)
(replace the checkpoint_path to yours in the .sh file)
cd LAFMA 
sh egs/tta/audiolfm/run_inference.sh
```
