"""
RMVPE — Robust Model for Vocal Pitch Estimation in Polyphonic Music
Adapted from RVC-Project/Retrieval-based-Voice-Conversion-WebUI

This is a standalone pitch extraction module. Only requires torch + numpy.
Model weights: models/rmvpe.pt (~140MB)
Download: https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
N_CLASS = 360
SAMPLE_RATE = 16000
HOP_LENGTH = 160  # 10ms per frame
N_MELS = 128
N_FFT = 1024
MEL_FMIN = 30
MEL_FMAX = 8000


def _to_local_average_cents(salience, center=None, thred=0.03):
    """Convert salience to cents using local average around peak."""
    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(salience * (np.arange(start, end) * 20 + 1997.3794084376191))
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if weight_sum > thred else 0
    if salience.ndim == 2:
        return np.array([_to_local_average_cents(s, thred=thred) for s in salience])
    raise ValueError(f"Expected 1D or 2D, got {salience.ndim}D")


def _cents_to_hz(cents):
    """Convert cents to Hz."""
    return np.where(cents > 0, 10 * 2 ** (cents / 1200), 0.0)


# ─── Mel Spectrogram ─────────────────────────────────────────────────────

class MelSpectrogram(nn.Module):
    def __init__(self, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 sample_rate=SAMPLE_RATE, fmin=MEL_FMIN, fmax=MEL_FMAX):
        super().__init__()
        import librosa
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax
        )
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('hann_window', torch.hann_window(n_fft))

    def forward(self, audio):
        # audio: (batch, samples)
        padding = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (padding, padding), mode='reflect')
        spec = torch.stft(
            audio, self.n_fft, self.hop_length,
            window=self.hann_window.to(audio.device),
            return_complex=True
        )
        mag = spec.abs()
        mel = torch.matmul(self.mel_basis.to(audio.device), mag)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel


# ─── BiGRU Block ──────────────────────────────────────────────────────────

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


# ─── ConvBlock (U-Net encoder/decoder) ────────────────────────────────────

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=momentum),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size,
                 n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(n_encoders):
            self.layers.append(
                nn.Sequential(
                    *[ConvBlockRes(out_channels if j > 0 else in_channels,
                                   out_channels, momentum)
                      for j in range(n_blocks)],
                    nn.Conv2d(out_channels, out_channels, kernel_size,
                              stride=(2, 2), padding=(kernel_size[0]//2, kernel_size[1]//2),
                              bias=False),
                    nn.BatchNorm2d(out_channels, momentum=momentum),
                    nn.ReLU(),
                )
            )
            self.latent_channels.append(out_channels)
            out_channels *= 2
        self.out_size = in_size // (2 ** n_encoders)
        self.out_channels = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t = x
            x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(*[ConvBlockRes(in_channels if j == 0 else out_channels,
                                         out_channels, momentum)
                            for j in range(n_blocks)])
        )
        for _ in range(n_inters - 1):
            self.layers.append(
                nn.Sequential(*[ConvBlockRes(out_channels, out_channels, momentum)
                                for _ in range(n_blocks)])
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(n_decoders):
            out_ch = in_channels // 2
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_ch, stride,
                                       stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch, momentum=momentum),
                    nn.ReLU(),
                    *[ConvBlockRes(out_ch * 2 if j == 0 else out_ch,
                                   out_ch, momentum)
                      for j in range(n_blocks)],
                )
            )
            in_channels = out_ch

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            # Upsample
            up = layer[0:3](x)
            # Crop/pad to match skip connection
            skip = concat_tensors[self.n_decoders - 1 - i]
            if up.shape != skip.shape:
                up = F.interpolate(up, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([up, skip], dim=1)
            x = layer[3:](x)
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size=(3, 3), n_blocks=2, en_de_layers=5,
                 inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size,
                               n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channels // 2,
            self.encoder.out_channels // 2,
            inter_layers, n_blocks
        )
        self.decoder = Decoder(
            self.encoder.out_channels // 2,
            en_de_layers, stride=(2, 2), n_blocks=n_blocks
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


# ─── E2E Model ────────────────────────────────────────────────────────────

class E2E(nn.Module):
    def __init__(self, n_blocks=2, n_gru=2, kernel_size=(3, 3),
                 en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers,
                             inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, N_CLASS),
            nn.Sigmoid(),
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel))
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


# ─── Main RMVPE Class ─────────────────────────────────────────────────────

class RMVPE:
    """
    RMVPE pitch estimator.

    Usage:
        model = RMVPE("models/rmvpe.pt", device="cpu")
        f0 = model.infer_from_audio("input.wav", sample_rate=16000)
        # f0: numpy array of Hz, 0 = unvoiced, 10ms per frame
    """

    def __init__(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'models', 'rmvpe.pt'
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RMVPE weights not found at {model_path}. "
                "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
            )

        self.device = device
        self.mel_extractor = MelSpectrogram().to(device)
        self.model = E2E(n_blocks=2, n_gru=2).to(device)

        # Load weights
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            ckpt = ckpt['model']
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()

    @torch.no_grad()
    def infer_from_audio(self, audio_input, sample_rate=16000, thred=0.03):
        """
        Extract F0 from audio.

        Args:
            audio_input: path to WAV file, or numpy array of audio samples
            sample_rate: sample rate of the audio (default 16000)
            thred: voicing threshold (lower = more voiced frames)

        Returns:
            numpy array of F0 in Hz, 0 = unvoiced, one value per 10ms
        """
        if isinstance(audio_input, str):
            import librosa
            audio_np, _ = librosa.load(audio_input, sr=SAMPLE_RATE, mono=True)
        else:
            audio_np = np.asarray(audio_input, dtype=np.float32)
            if sample_rate != SAMPLE_RATE:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

        # Process in chunks to avoid OOM
        chunk_size = SAMPLE_RATE * 30  # 30 seconds
        all_f0 = []

        for start in range(0, len(audio_np), chunk_size):
            chunk = audio_np[start:start + chunk_size]
            audio_t = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)

            mel = self.mel_extractor(audio_t)
            salience = self.model(mel).squeeze(0).cpu().numpy()

            # Convert salience to cents then to Hz
            cents = _to_local_average_cents(salience, thred=thred)
            f0_chunk = _cents_to_hz(cents)
            all_f0.append(f0_chunk)

        return np.concatenate(all_f0)
