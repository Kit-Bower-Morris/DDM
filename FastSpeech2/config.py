from dataclasses import dataclass, field
from typing import List
from trainer import TrainerConfig

@dataclass
class Fastspeech2Config(TrainerConfig):

    #dataset format
    dataset: str = "LJSpeech"
    
    #data paths
    preprocessed_path: str = "D:/Chimera/Chopra/dataset/data/preprocessed_data/"
    corpus_path: str = "/home/ming/Data/LJSpeech-1.1"
    lexicon_path: str = "lexicon/librispeech-lexicon.txt"
    output_path: str = "./output/"
    raw_path: str = "./raw_data/LJSpeech"
    ckpt_path: str = "./output/ckpt/LJSpeech"
    log_path: str = "./output/log/LJSpeech"
    result_path: str = "./output/result/LJSpeech"
    
    #training
    batch_size: int = 16
    multi_speaker: bool = False
    dashboard_logger: str = "tensorboard"
    epochs: int = 20
    print_step: int = 5
    save_step: int = 5
    plot_step: int = 5
    data_path: List[str] = field(default_factory=lambda:["train.txt", "val.txt"])
    max_seq_len: int = 1000
    mixed_precision: bool = True #if this goes wrong set to false
    precision: str = "fp16" #and set this to fp32


    #steps
    total_step: int = 900000
    log_model_step: int = 100
    synth_step: int = 1000
    val_step: int = 1000
    save_step: int = 100000

    #optimizer
    '''
    betas: List[float] = field(default_factory=lambda:[0.9, 0.98])
    eps: float = 1e-9
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_acc_step: int = 1
    warm_up_step: int = 4000
    anneal_steps: List[int] = field(default_factory=lambda:[300000, 400000, 500000])
    anneal_rate: float = 0.3
    '''
    optimizer: str = "Adam"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.9, 0.98], "weight_decay": 0.0, "eps": 1e-9})
    lr_scheduler: str = "NoamLR"
    lr_scheduler_params: dict = field(default_factory=lambda: {"warmup_steps": 4000})
    scheduler_after_epoch: bool = False
    lr: float = 1e-4
    grad_clip: float = 1.0

    #transformer
    encoder_layer: int = 4
    encoder_head: int = 2
    encoder_hidden: int = 256
    decoder_layer: int = 6
    decoder_head: int = 2
    decoder_hidden: int = 256
    conv_filter_size: int = 1024
    conv_kernel_size: List[int] = field(default_factory=lambda:[9, 1])
    encoder_dropout: float = 0.2
    decoder_dropout: float = 0.2

    #variance predictor
    filter_size: int = 256
    kernel_size: int = 3
    dropout: float = 0.5

    #variance embedding
    pitch_quantization: str = "linear" # support 'linear' or 'log', 'log' is allowed only if the pitch values are not normalized during preprocessing
    energy_quantization: str = "linear" # support 'linear' or 'log', 'log' is allowed only if the energy values are not normalized during preprocessing
    n_bins: int = 256

    #gst
    conv_filters: List[int] = field(default_factory=lambda:[32, 32, 64, 64, 128, 128])
    n_style_token: int = 10
    attn_head: int = 4
    num_mel: int = 80
    gst_embedding_dim: int = 256

    #vocoder
    model: str = "HiFi-GAN" # support 'HiFi-GAN', 'MelGAN'
    speaker: str = "universal" # support  'LJSpeech', 'universal'

    #data
    val_size: int = 512
    sampling_rate: int = 22050
    max_wav_value: float = 32768.0
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: int = 0
    mel_fmax: int = 8000

    #text
    text_cleaners: List[str] = field(default_factory=lambda:["english_cleaners"])
    language: str = "en"

    #pitch and energy
    feature: str = "phoneme_level" # support 'phoneme_level' or 'frame_level'
    normalization: bool = True

