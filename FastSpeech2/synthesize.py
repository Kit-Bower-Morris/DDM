import re
import argparse
from string import punctuation
import os

import torch

import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.tools import to_device, synth_samples, pad_1D, pad_2D

from dataset import TextDataset
from text import text_to_sequence
from model.fastspeech2 import FastSpeech2
from config import Fastspeech2Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(config["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, config["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, config, batch):
    from utils.model import get_vocoder
    vocoder = get_vocoder(config, device)


    with torch.no_grad():
        # Forward
        model.inference_step(batch, vocoder)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--restore_step", type=int)
    parser.add_argument("--model_path", type=str, required=True)
    #parser.add_argument("--config_path", type=str, required=True)

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize",
    )

    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )

    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    config = Fastspeech2Config()

    # Get model
    
    model = FastSpeech2(config).to(device)
 
    ckpt = torch.load(args.model_path)

    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.requires_grad_ = False


    ids = raw_texts = [args.text[:100]]
    speakers = np.array([args.speaker_id])
    texts = np.array([preprocess_english(args.text, config)])
    text_lens = np.array([len(texts[0])])
    mel_path = 'Chopra-mel-01_chunk5_split1.npy'
    mels = np.load(mel_path)
    mels = np.expand_dims(mels, axis=0)
    mel_lens = np.array([mel.shape[0] for mel in mels])
    pitches, energies, durations = args.pitch_control, args.energy_control, args.duration_control
    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])
    speakers = np.array(speakers)
    #texts = pad_1D(texts)
    #mels = pad_2D(mels)
    #pitches = pad_1D(pitches)
    #energies = pad_1D(energies)
    #durations = pad_1D(durations)
    speakers = torch.from_numpy(speakers).long().to(device)
    texts = torch.from_numpy(texts).long().to(device)
    text_lens = torch.from_numpy(text_lens).to(device)
    mels = torch.from_numpy(mels).float().to(device)
    #mels = torch.unsqueeze(mels, dim=0)
    mel_lens = torch.from_numpy(mel_lens).to(device)
    #pitches = torch.from_numpy(pitches).float().to(device)
    #energies = torch.from_numpy(energies).to(device)
    #durations = torch.from_numpy(durations).long().to(device)
    batch =  {
        'ids': ids,
        'raw_texts': raw_texts,
        'speakers': speakers,
        'texts': texts,
        'text_lens': text_lens,
        'max_text_lens': max(text_lens),
        'mels': mels,
        'mel_lens': mel_lens,
        'max_mel_lens': max(mel_lens),
        'pitches': pitches,
        'energies': energies,
        'durations': durations,
    }


    synthesize(model, config, batch)
