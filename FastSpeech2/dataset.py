import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D
#from mel_check import test
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    def __init__(
        self, filename, config, device, sort=False, drop_last=False
    ):
        self.device = device
        self.dataset_name = config["dataset"]
        self.preprocessed_path = config["preprocessed_path"]
        self.cleaners = config["text_cleaners"]
        self.batch_size = config["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]


        #text_lens = torch.tensor(np.array([text.shape[0] for text in texts]))
        #mel_lens = torch.tensor(np.array([mel.shape[0] for mel in mels]))
        
        #speakers = np.array(speakers)
        #texts = torch.tensor(pad_1D(texts))
        #mels = torch.tensor(pad_2D(mels))
        #pitches = torch.tensor(pad_1D(pitches)).to(torch.float32)
        #energies = torch.tensor(pad_1D(energies))
        #durations = torch.tensor(pad_1D(durations))

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        speakers = torch.from_numpy(speakers).long().to(self.device)
        texts = torch.from_numpy(texts).long().to(self.device)
        text_lens = torch.from_numpy(text_lens).to(self.device)
        mels = torch.from_numpy(mels).float().to(self.device)
        mel_lens = torch.from_numpy(mel_lens).to(self.device)
        pitches = torch.from_numpy(pitches).float().to(self.device)
        energies = torch.from_numpy(energies).to(self.device)
        durations = torch.from_numpy(durations).long().to(self.device)

        #test(mels, mel_lens)

        return {
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

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        idx_arr = [idx_arr.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output[0]


class TextDataset(Dataset):
    def __init__(self, filepath, config):
        self.cleaners = config["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                config["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


