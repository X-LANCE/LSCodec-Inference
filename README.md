# LSCodec: Low-Bitrate and Speaker-Decoupled Discrete Speech Codec
> This is the inference checkpoint and code of LSCodec.

[![paper](https://img.shields.io/badge/paper-arxiv:2410.15764-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.15764)
[![demo](https://img.shields.io/badge/demo-page-green)](https://cantabile-kwok.github.io/LSCodec/)

## Environment
Our code is tested on Python 3.10. Please use the `requirements.txt`:
```bash
conda create -n lscodec python=3.10
conda activate lscodec
pip install -r requirements.txt
```
Or, if your prefer Docker, you can directly use the image from vec2wav 2.0:
```bash
docker pull cantabilekwok511/vec2wav2.0:v0.2
docker run -it -v /path/to/vec2wav2.0:/workspace cantabilekwok511/vec2wav2.0:v0.2
```

## Checkpoints
Checkpoints can be downloaded from [this path](). You can use this script to automatically download them:
```
bash download_ckpt.sh
```
This will create `pretrained/` and download the following files:

* `codebook.npy`: the codebook (1, 300, 64) representing the codebook of LSCodec-50Hz.
* `encoder_config.yml`, `vocoder_config.yml`: configs for the encoder and vocoder, respectively.
* `lscodec_encoder.pt`, `lscodec_vocoder.pt`: checkpoints for the encoder and vocoder, respectively.

This downloading script will also prompt you to download the WavLM checkpoint manually. Please put this file under `pretrained/` as well.
* `WavLM-Large.pt`: WavLM-Large checkpoint from the [official repo](https://github.com/microsoft/unilm/blob/master/wavlm/README.md).

## Encoding Waveform to Tokens
This codebase uses `kaldiio` to load and store data.
Firstly, please prepare a `wav.scp` file containing the wav files:
```
utt-1 /path/to/utt_1.wav
utt-2 /path/to/utt_2.wav
...
```
You can also refer to `example/wav.scp` for example.

Then, encoding can be done by
```bash
source path.sh
encode.py --wav-scp example/wav.scp \
          --outdir example/tokens/
```
where the tokens are stored in `example/tokens/feats.ark` and `feats.scp`. The `feats.scp` should look like:
```
3570_5694_000009_000002 /path/to/example/tokens/feats.ark:24
8455_210777_000079_000002 /path/to/example/tokens/feats.ark:677
```
You can also look into `lscodec/bin/encode.py` if you want to save into different formats.

## Vocoding with Reference Prompts
Once encoded, LSCodec tokens can be vocoded into 24kHz waveforms using
```bash
source path.sh
decode_wav_prompt.py --feats-scp example/tokens/feats.scp \
    --prompt-wav-scp example/prompt.scp \
    --outdir example/wav
```
where `--prompt-wav-scp prompt.scp` specifies the prompt wav for each utterance's token sequence. This `prompt.scp` looks like:
```
utt-1 /path/to/reference_utt_1.wav
utt-2 /path/to/reference_utt_2.wav
```
Finally, the decoded waveforms can be found in `example/wav`.

## Combining Encoding and Vocoding into One Step
If you want to use one script for the encoding and vocoding process together, consider:
```bash
source path.sh
recon_with_prompt.py --wav-scp example/wav.scp \
    --prompt-wav-scp example/prompt.scp \
    --outdir example/wav
```

## Citation
```bibtex
@article{guo2024lscodec,
	author={Yiwei Guo and Zhihan Li and Chenpeng Du and Hankun Wang and Xie Chen and Kai Yu},
	title={{LSCodec}: Low-Bitrate and Speaker-Decoupled Discrete Speech Codec},
	journal={arXiv preprint arXiv:2410.15764},
	year={2024},
}
```
