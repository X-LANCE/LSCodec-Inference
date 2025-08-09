# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# Copyright 2025 Yiwei Guo
# Portions of this code are a modification of the ParallelWaveGAN project (https://github.com/kan-bayashi/ParallelWaveGAN).
# Licensed under the MIT license.

"""Utility functions."""

import fnmatch
import logging
import os
import sys
import tarfile

from distutils.version import LooseVersion
from filelock import FileLock

import h5py
import numpy as np
import torch
import soundfile as sf
import torchaudio.transforms as transforms
import yaml
import re
import kaldiio

def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data

def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()

class HDF5ScpLoader(object):
    """Loader class for a fests.scp file of hdf5 file.

    Examples:
        key1 /some/path/a.h5:feats
        key2 /some/path/b.h5:feats
        key3 /some/path/c.h5:feats
        key4 /some/path/d.h5:feats
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp")
        >>> array = loader["key1"]

        key1 /some/path/a.h5
        key2 /some/path/b.h5
        key3 /some/path/c.h5
        key4 /some/path/d.h5
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp", "feats")
        >>> array = loader["key1"]

        key1 /some/path/a.h5:feats_1,feats_2
        key2 /some/path/b.h5:feats_1,feats_2
        key3 /some/path/c.h5:feats_1,feats_2
        key4 /some/path/d.h5:feats_1,feats_2
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp")
        # feats_1 and feats_2 will be concatenated
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp, default_hdf5_path="feats"):
        """Initialize HDF5 scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with hdf5 format.
            default_hdf5_path (str): Path in hdf5 file. If the scp contain the info, not used.

        """
        self.default_hdf5_path = default_hdf5_path
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get hdf5 file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        p = self.data[key]
        if ":" in p:
            if len(p.split(",")) == 1:
                return read_hdf5(*p.split(":"))
            else:
                p1, p2 = p.split(":")
                feats = [read_hdf5(p1, p) for p in p2.split(",")]
                return np.concatenate(
                    [f if len(f.shape) != 1 else f.reshape(-1, 1) for f in feats], 1
                )
        else:
            return read_hdf5(p, self.default_hdf5_path)

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]

class NpyScpLoader(object):
    """Loader class for a fests.scp file of npy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...
        >>> loader = NpyScpLoader("feats.scp")
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp):
        """Initialize npy scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with npy format.

        """
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get npy file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        return np.load(self.data[key])

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]

def load_model(config=None, checkpoint=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.

    Return:
        torch.nn.Module: Model instance.

    """
    assert (config is not None) or (checkpoint is not None)
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    from lscodec.models.feature_extractor import ConvFeatureExtractionModel
    from lscodec.models.feature_extractor import ConvAggregator
    from lscodec.models.lscodec import LSCodecEncoderQuantizer, LSCodecEncoder
    import lscodec
    from lscodec.models.quantization.core_vq import GroupVectorQuantization
    from lscodec.models.fairseq_modules.gumbel_vector_quantizer import GumbelVectorQuantizer
    feature_extractor = ConvFeatureExtractionModel(
        conv_layers=eval(config['feature_extractor_params']["conv_feature_layers"]),
        dropout=0.0,
        log_compression=config['feature_extractor_params']['log_compression'],
        skip_connections=config['feature_extractor_params']['skip_connections_feat'],  # False
        residual_scale=config['feature_extractor_params']['residual_scale'],  # 0.25
        non_affine_group_norm=config['feature_extractor_params']['non_affine_group_norm'],  # False
        activation=getattr(torch.nn, config['feature_extractor_params']['activation'])()  # ReLU
    )
    embed_dim = eval(config['feature_extractor_params']["conv_feature_layers"])[-1][0]
    feature_aggregator = ConvAggregator(
        conv_layers=eval(config['feature_aggregator_params']['conv_feature_layers']),
        dropout=0.0,
        embed=embed_dim,
        skip_connections=config['feature_aggregator_params']['skip_connections'],
        residual_scale=config['feature_aggregator_params']['residual_scale'],
        non_affine_group_norm=config['feature_aggregator_params']['non_affine_group_norm'],
        conv_bias=config['feature_aggregator_params']['conv_bias'],
        zero_pad=config['feature_aggregator_params']['zero_pad'],
        activation=getattr(torch.nn, config['feature_aggregator_params']['activation'])()
    )
    embed_dim = eval(config['feature_aggregator_params']["conv_feature_layers"])[-1][0] // 2  # NOTE: divide by 2 because of mean and logvar
    conv_transformer_frontend = LSCodecEncoder(
        conv_extraction_model=feature_extractor,
        conv_aggregation_model=feature_aggregator,
    )
    if config['have_quantizer']:
        if config['quantizer_class'] == "KMeans":
            vq_dim = config['quantizer_params']['vq_dim'] // config['quantizer_params']['vq_groups']
            quantizer = GroupVectorQuantization(
                num_quantizers=config['quantizer_params']['vq_groups'],
                dim=embed_dim // config['quantizer_params']['vq_groups'],
                codebook_size=config['quantizer_params']['vq_size_each_group'],
                codebook_dim=vq_dim,
                decay=config['quantizer_params']['decay'],
                epsilon=config['quantizer_params']['epsilon'],
                kmeans_init=config['quantizer_params']['kmeans_init'],
                kmeans_iters=config['quantizer_params']['kmeans_iters'],
                threshold_ema_dead_code=config['quantizer_params']['threshold_ema_dead_code'],
                commitment_weight=config['quantizer_params']['commitment_weight'],
                init_codebook=config.get("pretrain_codebook", None)
            )
        elif config['quantizer_class'] == "Gumbel":
            quantizer = GumbelVectorQuantizer(
                dim=embed_dim,
                num_vars=config['quantizer_params']['vq_size_each_group'],
                groups=config['quantizer_params']['vq_groups'],
                temp=config['quantizer_params']['temp'],
                combine_groups=config['quantizer_params']['combine_groups'],
                vq_dim=config['quantizer_params']['vq_dim'],
                time_first=False,
                activation=getattr(torch.nn,
                                   config['quantizer_params'].get("activation", "GELU"))(),
                weight_proj_depth=config['quantizer_params'].get("weight_proj_depth", 1),
                weight_proj_factor=config['quantizer_params'].get("weight_proj_factor", 1)
            )
        else:
            raise NotImplementedError(f"{config['quantizer_class']} is not implemented!")
    else:
        quantizer = None

    model = LSCodecEncoderQuantizer(
            frontend=conv_transformer_frontend,
            quantizer=quantizer,
            dropout_features=config['dropout_features'],
            mean_only=config.get("mean_only", False),
        )
    if checkpoint is not None:
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu")["model"]["generator"]
        )

    return model

def crop_seq(x, offsets, length):
    """Crop padded tensor with specified length.

    :param x: (torch.Tensor) The shape is (B, C, D)
    :param offsets: (list)
    :param min_len: (int)
    :return:
    """
    B, C, T = x.shape
    x_ = x.new_zeros(B, C, length)
    for i in range(B):
        x_[i, :, :min(length, T-offsets[i])] = x[i, :, offsets[i]: offsets[i] + length]
    return x_

def read_wav_16k(audio_path):
    """Process audio file to 16kHz sample rate"""
    if isinstance(audio_path, tuple):  # Gradio audio input returns (sample_rate, audio_data)
        wav = audio_path[1]
        sr = audio_path[0]
    else:  # Regular file path
        assert os.path.exists(audio_path), f"File not found: {audio_path}"
        wav, sr = sf.read(audio_path)

    if sr != 16000:
        audio_tensor = torch.tensor(wav, dtype=torch.float32)
        resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(audio_tensor)
        wav = wav.numpy()
    return wav

def load_vocoder(config=None, checkpoint=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.

    Return:
        torch.nn.Module: Model instance.

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import lscodec.models.ctx_v2w as ctx_vec2wav

    # get model and load parameters
    model_class = getattr(
        ctx_vec2wav,
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    model = ctx_vec2wav.CTXVEC2WAVGenerator(
        ctx_vec2wav.CTXVEC2WAVFrontend(config["prompt_net_type"], config["num_mels"], **config["frontend_params"]),
        model_class(**config["generator_params"])
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    # add pqmf if needed
    if config["generator_params"]["out_channels"] > 1:
        # lazy load for circular error
        from lscodec.layers import PQMF

        pqmf_params = {}
        if LooseVersion(config.get("version", "0.1.0")) <= LooseVersion("0.4.2"):
            # For compatibility, here we set default values in version <= 0.4.2
            pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.backend.pqmf = PQMF(
            subbands=config["generator_params"]["out_channels"],
            **config.get("pqmf_params", pqmf_params),
        )

    return model

def load_audio(audio_path):
    if re.search(r'ark:\d+$', audio_path):
        sr, audio = kaldiio.load_mat(audio_path)
    else:
        audio, sr = sf.read(audio_path)
    return audio, sr
