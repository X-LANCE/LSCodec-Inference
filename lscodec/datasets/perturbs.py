# -*- coding: utf-8 -*-

# Portions of this code are based on ContentVec.
# The original source code can be found at:
# https://github.com/auspicious3000/contentvec/blob/d746688a32940f4bee410ed7c87ec9cf8ff04f74/contentvec/data/audio/audio_utils_1.py
#
# Copyright (c) 2025 Yiwei Guo
# Licensed under the MIT license.

import os
import numpy as np
from scipy.signal import sosfilt
from scipy import signal as sg
import sox


def make_lowshelf(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Lowshelf filter.
    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 
    https://www.w3.org/2011/audio/audio-eq-cookbook.html
    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.
    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    aminus1 = A - 1
    aplus1 = A + 1
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    coso = np.cos(omega)
    beta = np.sin(omega) * np.sqrt(A) / Q 
    aminus1TimesCoso = aminus1 * coso

    # coefs calculation
    b0 = A * (aplus1 - aminus1TimesCoso + beta)
    b1 = A * 2 * (aminus1 - aplus1 * coso)
    b2 = A * (aplus1 - aminus1TimesCoso - beta)
    a0 = aplus1 + aminus1TimesCoso + beta
    a1 = -2 * (aminus1 + aplus1 * coso)
    a2 = aplus1 + aminus1TimesCoso - beta

    # output coefs 
    #b = np.array([b0/a0, b1/a0, b2/a0])
    #a = np.array([a0/a0, a1/a0, a2/a0])

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])



def make_highself(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Highshelf filter.
    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 
    https://www.w3.org/2011/audio/audio-eq-cookbook.html
    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.
    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    aminus1 = A - 1
    aplus1 = A + 1
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    coso = np.cos(omega)
    beta = np.sin(omega) * np.sqrt(A) / Q 
    aminus1TimesCoso = aminus1 * coso

    # coefs calculation
    b0 = A * (aplus1 + aminus1TimesCoso + beta)
    b1 = A * -2 * (aminus1 + aplus1 * coso)
    b2 = A * (aplus1 + aminus1TimesCoso - beta)
    a0 = aplus1 - aminus1TimesCoso + beta
    a1 = 2 * (aminus1 - aplus1 * coso)
    a2 = aplus1 - aminus1TimesCoso - beta

    # output coefs
    #b = np.array([b0/a0, b1/a0, b2/a0])
    #a = np.array([a0/a0, a1/a0, a2/a0])
      
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])



def make_peaking(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Peaking EQ.
    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 
    https://www.w3.org/2011/audio/audio-eq-cookbook.html
    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.
    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    alpha = np.sin(omega) / (Q * 2)
    c2 = -2 * np.cos(omega)
    alphaTimesA = alpha * A
    alphaOverA = alpha / A

    # coefs calculation
    b0 = 1 + alphaTimesA
    b1 = c2
    b2 = 1 - alphaTimesA
    a0 = 1 + alphaOverA
    a1 = c2
    a2 = 1 - alphaOverA

    # output coefs
    #b = np.array([b0/a0, b1/a0, b2/a0])
    #a = np.array([a0/a0, a1/a0, a2/a0])
    
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])



def params2sos(G, Fc, Q, fs):
    """Convert 5 band EQ paramaters to 2nd order sections.
    Takes a vector with shape (13,) of denormalized EQ parameters
    and calculates filter coefficients for each of the 5 filters.
    These coefficients (2nd order sections) are then stored into a
    single (5,6) matrix. This matrix can be fed to `scipy.signal.sosfreqz()`
    in order to determine the frequency response of the cascasd of
    all five biquad filters.
    Args:
        x  (float): Gain factor in dB.       
        fs (float): Sampling frequency in Hz.
    Returns:
        ndarray: filter coefficients for 5 band EQ stored in (5,6) matrix.
        [[b1_0, b1_1, b1_2, a1_0, a1_1, a1_2],  # lowshelf coefficients
         [b2_0, b2_1, b2_2, a2_0, a2_1, a2_2],  # first band coefficients
         [b3_0, b3_1, b3_2, a3_0, a3_1, a3_2],  # second band coefficients
         [b4_0, b4_1, b4_2, a4_0, a4_1, a4_2],  # third band coefficients
         [b5_0, b5_1, b5_2, a5_0, a5_1, a5_2]]  # highshelf coefficients
    """
    # generate filter coefficients from eq params
    c0 = make_lowshelf(G[0], Fc[0], Q[0], fs=fs)
    c1 = make_peaking (G[1], Fc[1], Q[1], fs=fs)
    c2 = make_peaking (G[2], Fc[2], Q[2], fs=fs)
    c3 = make_peaking (G[3], Fc[3], Q[3], fs=fs)
    c4 = make_peaking (G[4], Fc[4], Q[4], fs=fs)
    c5 = make_peaking (G[5], Fc[5], Q[5], fs=fs)
    c6 = make_peaking (G[6], Fc[6], Q[6], fs=fs)
    c7 = make_peaking (G[7], Fc[7], Q[7], fs=fs)
    c8 = make_peaking (G[8], Fc[8], Q[8], fs=fs)
    c9 = make_highself(G[9], Fc[9], Q[9], fs=fs)

    # stuff coefficients into second order sections structure
    sos = np.concatenate([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9], axis=0)

    return sos


import parselmouth
def change_gender(x, fs, lo, hi, ratio_fs, ratio_ps, ratio_pr):
    s = parselmouth.Sound(x, sampling_frequency=fs)
    f0 = s.to_pitch_ac(pitch_floor=lo, pitch_ceiling=hi, time_step=0.8/lo)
    f0_np = f0.selected_array['frequency']
    f0_med = np.median(f0_np[f0_np!=0]).item()
    ss = parselmouth.praat.call([s, f0], "Change gender", ratio_fs, f0_med*ratio_ps, ratio_pr, 1.0)
    return ss.values.squeeze(0)

def change_gender_f0(x, fs, lo, hi, ratio_fs, new_f0_med, ratio_pr):
    s = parselmouth.Sound(x, sampling_frequency=fs)
    ss = parselmouth.praat.call(s, "Change gender", lo, hi, ratio_fs, new_f0_med, ratio_pr, 1.0)
    return ss.values.squeeze(0)

def parse_range(range_str):
    # (a,b)|(c,d)|p
    
    terms = range_str.split("|")
    assert len(terms)==3, range_str
    prob = float(terms[-1])
    ranges = [eval(terms[0]), eval(terms[1])]
    return ranges, prob

class FormantF0Perturb:
    def __init__(self, male_formant_range, female_formant_range, female_f0_range):
        # male_formant_range: (0.9,0.95)|(1.05,1.3)|0.25
        # first range, second range, prob of first range
        
        self.Qmin = 2
        self.Qmax = 5
        self.male_formant_range, self.male_formant_prob = parse_range(male_formant_range)
        self.female_formant_range, self.female_formant_prob = parse_range(female_formant_range)
        self.female_f0_range, self.female_f0_prob = parse_range(female_f0_range)
        
    def get_perturb_param(self, gender):
        # formant shift and pitch shift.
        if gender == "M":
            if np.random.random()<=self.male_formant_prob:
                ratio_fs = np.random.uniform(self.male_formant_range[0][0], self.male_formant_range[0][1])
            else:
                ratio_fs = np.random.uniform(self.male_formant_range[1][0], self.male_formant_range[1][1])
            return 75, 250, ratio_fs, 1.0
        else:
            if np.random.random()<=self.female_formant_prob:
                ratio_fs = np.random.uniform(self.female_formant_range[0][0], self.female_formant_range[0][1])
            else:
                ratio_fs = np.random.uniform(self.female_formant_range[1][0], self.female_formant_range[1][1])
            if np.random.random()<=self.female_f0_prob:
                ratio_ps = np.random.uniform(self.female_f0_range[0][0], self.female_f0_range[0][1])
            else:
                ratio_ps = np.random.uniform(self.female_f0_range[1][0], self.female_f0_range[1][1])
            return 100, 400, ratio_fs, ratio_ps

    def random_eq(self, wav, sr):
        Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))
        # rng = np.random.default_rng()
        z = np.random.uniform(0, 1, size=(10,))
        Q = self.Qmin * (self.Qmax / self.Qmin)**z
        G = np.random.uniform(-12, 12, size=(10,))
        sos = params2sos(G, Fc, Q, sr)
        wav = sosfilt(sos, wav)
        return wav

    def __call__(self, wav, sr, gender):
        # gender: "F" or "M"
        
        lo, hi, ratio_fs, ratio_ps = self.get_perturb_param(gender)
        # fs: formant shift. This controls timbre (higher -> more female). ps: pitch shift. This controls absolute pitch level
        # We fix the pitch range parameter to 1. So that we don't change the pitch variations
        ratio_pr = 1
        
        ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)
        
        return ss

class SpeedPerturb:
    def __init__(self, male_range, female_range):
        self.male_range, self.male_prob = parse_range(male_range)
        self.female_range, self.female_prob = parse_range(female_range)
        
    def __call__(self, wav, sr, gender):
        
        if gender == "F":
            ranges, prob = self.male_range, self.male_prob
        else:
            ranges, prob = self.female_range, self.female_prob
        
        if np.random.random()<=prob:
            speed_alpha = np.random.uniform(ranges[0][0], ranges[0][1])
        else:
            speed_alpha = np.random.uniform(ranges[1][0], ranges[1][1])
        
        self.transformer = sox.Transformer()
        self.transformer.set_globals(verbosity=1)  # disable warnings.
        self.transformer.speed(factor=speed_alpha)
        self.transformer.tempo(factor=1/speed_alpha)  # recover the original length. Only affects timbre and pitch.
        wav_perturb = self.transformer.build_array(input_array=wav, sample_rate_in=sr)
        return wav_perturb

class IdentityPerturb:
    def __init__(self, ):
        pass
    def __call__(self,wav, sr, gender):
        return wav
        
class Perturbs:
    def __init__(self, formant_f0_perturb_params, speed_perturb_params, speed_perturb_prob):
        self.speed_perturb_prob = speed_perturb_prob
        if speed_perturb_prob!=0.0:
            self.speed_perturber = SpeedPerturb(**speed_perturb_params)
        else:
            self.speed_perturber = IdentityPerturb()
        if speed_perturb_prob!=1.0:
            self.formant_perturber = FormantF0Perturb(**formant_f0_perturb_params)
        else:
            self.formant_perturber = IdentityPerturb()
    def __call__(self, wav, sr, gender):
        if np.random.random()<=self.speed_perturb_prob:
            return self.speed_perturber(wav,sr,gender)
        else:
            return self.formant_perturber(wav,sr,gender)