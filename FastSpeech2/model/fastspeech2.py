import os
import json
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp.autocast_mode import autocast

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from .gst import GST
from utils.tools import get_mask_from_lengths, synth_one_sample
from .loss import FastSpeech2Loss
from dataset import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, config):
        super(FastSpeech2, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.gst_layer = GST(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)
        self.mel_linear = nn.Linear(
            config.decoder_hidden,
            config.num_mel,
        )
        self.postnet = PostNet()
        

        self.speaker_emb = None
        if config.multi_speaker:
            with open(
                os.path.join(
                    config.preprocessed_path, "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                config.encoder_hidden,
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks)
        gst_output = self.gst_layer(mels) 
        gst_output_ = gst_output.expand(output.size(0), output.size(1), -1)
        output = output + gst_output_



        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        
        
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        output, mel_masks = self.decoder(output, mel_masks)
        
        output = self.mel_linear(output)
        output = output.to(torch.float32)
        postnet_output = self.postnet(output) + output

        return {
            'output': output,
            'postnet_output': postnet_output,
            'p_predictions': p_predictions,
            'e_predictions': e_predictions,
            'log_d_predictions': log_d_predictions,
            'd_rounded': d_rounded,
            'src_masks': src_masks,
            'mel_masks': mel_masks,
            'src_lens': src_lens,
            'mel_lens': mel_lens,
        }


    def train_step(self, batch, criterion):
        speakers = batch['speakers']
        texts = batch['texts']
        text_lens = batch['text_lens']
        max_text_lens = batch['max_text_lens']
        mels = batch['mels']
        mel_lens = batch['mel_lens']
        max_mel_lens = batch['max_mel_lens']
        pitches = batch['pitches']
        energies = batch['energies']
        durations = batch['durations']
        with autocast(enabled=True):
            output = self.forward(
                speakers,
                texts,
                text_lens,
                max_text_lens,
                mels,
                mel_lens,
                max_mel_lens,
                pitches,
                energies,
                durations
                )
        #self._create_logs(self, batch, output)
        losses = criterion(batch, output)
        return output, losses

    @torch.no_grad()
    def eval_step(self, batch, criterion):
        return self.train_step(batch, criterion)
    

    def get_criterion(self):
        return FastSpeech2Loss(self.config).to(device)
    
    
    def get_train_data_loader(
        self, config, assets, samples, verbose, num_gpus
    ):  
        dataset = Dataset(
            "train.txt", config, device, sort=False, drop_last=True
        )
        batch_size = config.batch_size

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        return loader    

    
    def get_eval_data_loader(
        self, config, assets, samples, verbose, num_gpus
    ):  
        dataset = Dataset(
            "val.txt", config, device, sort=True, drop_last=True
        )
        batch_size = config.batch_size
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        return loader    
    
    def format_batch(self, batch: Dict) -> Dict:
        return batch
    
    def format_batch_on_device(self, batch):
        return batch
    
    def train_log(
            self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
            ):
        figures, audios = self._create_logs(batch, outputs)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.config.sampling_rate)


    def eval_log(
            self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
            ):
        figures, audios = self._create_logs(batch, outputs)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.config.sampling_rate)
    

    def _create_logs(self, batch, outputs):
        from utils.model import get_vocoder
        vocoder = get_vocoder(self.config, device)
        # Sample audio
        figures, wav_reconstruction, synthsised_wav, tag = synth_one_sample(
                batch,
                outputs,
                vocoder,
                self.config
            )
        wav_reconstruction = wav_reconstruction[0].squeeze(0).cpu().numpy()
        synthsised_wav = synthsised_wav[0].squeeze(0).detach().cpu().numpy()
        return figures, {"synthsised_audio": synthsised_wav, "reconstruction_audio": wav_reconstruction}
    
