"""Numerical FT adapter; compact architecture from the project benchmark.
Reference: https://github.com/yandex-research/rtdl-revisiting-models
"""
import io
import math
import os
import random
import time
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data import Standardizer, fit_standardizer
from metrics import participant_macro_f1

N_FEATURES = 135
CLASSES = ('Sitting', 'Standing', 'Lying_Down', 'Walking')


@dataclass(frozen=True)
class FTConfig:
    d_token: int = 64
    n_blocks: int = 2
    n_heads: int = 4
    ffn_hidden: int = 128
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 8
    weight_decay: float = 1e-5
    seed: int = 17
    amp: bool = True
    deterministic: bool = True

    def __post_init__(self):
        for name in ('d_token', 'n_blocks', 'n_heads', 'ffn_hidden', 'batch_size', 'max_epochs', 'patience'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.d_token % self.n_heads:
            raise ValueError('d_token must be divisible by n_heads')
        for name in ('attention_dropout', 'ffn_dropout'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f'{name} must be in [0, 1)')
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if type(self.amp) is not bool or type(self.deterministic) is not bool:
            raise ValueError('amp and deterministic must be booleans')


def configure_runtime(cfg, *, training):
    """Also called on checkpoint-only resumes; never reseeds during prediction."""
    if cfg.deterministic:
        # Set before the first CUDA tensor/operation, including checkpoint loading.
        workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if workspace not in (':4096:8', ':16:8'):
            raise ValueError('Deterministic FT requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
    torch.use_deterministic_algorithms(cfg.deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.amp and (training or device.type == 'cuda'):
        if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            raise ValueError('amp=True requires a CUDA GPU supporting BF16; use the CUDA desktop for the formal run')
    return device


def runtime_details(device, cfg):
    return {
        'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'gpu_capability': list(torch.cuda.get_device_capability(device)) if device.type == 'cuda' else None,
        'precision': 'bf16_autocast' if cfg.amp and device.type == 'cuda' else 'float32',
        'deterministic': cfg.deterministic, 'tf32': False,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cpu_threads': torch.get_num_threads(),
    }


def checked_features(X, *, allow_empty=False):
    values = np.asarray(X, dtype=np.float32)
    if (values.ndim != 2 or values.shape[1] != N_FEATURES
            or (not allow_empty and not len(values)) or not np.isfinite(values).all()):
        raise ValueError('Expected finite features with shape (n, 135) and nonempty training/validation data')
    return values


def check_table(table, *, training):
    X = checked_features(table.X)
    y = np.asarray(table.y)
    if (y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any() or (y >= len(CLASSES)).any()):
        raise ValueError('Expected one integer class ID (0..3) per feature row')
    if np.asarray(table.participant_id).shape != y.shape:
        raise ValueError('Expected one participant ID per feature row')
    if training and not np.array_equal(np.unique(y), np.arange(len(CLASSES))):
        raise ValueError('All four classes must be present in the current training fold')


class Block(nn.Module):
    def __init__(self, cfg, first):
        super().__init__()
        d = cfg.d_token
        self.heads = cfg.n_heads
        self.attention_dropout = cfg.attention_dropout
        self.norm1 = nn.Identity() if first else nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 2 * cfg.ffn_hidden)
        self.ff2 = nn.Linear(cfg.ffn_hidden, d)
        self.drop = nn.Dropout(cfg.ffn_dropout)

    def forward(self, x, last=False):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if last:
            q = q[:, :, :1]
            x = x[:, :1]
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attention_dropout if self.training else 0.)
        x = x + self.proj(a.transpose(1, 2).reshape(b, x.shape[1], d))
        a, gate = self.ff1(self.norm2(x)).chunk(2, dim=-1)
        return x + self.ff2(self.drop(a * F.relu(gate)))


class FTTransformer(nn.Module):
    def __init__(self, cfg=FTConfig()):
        super().__init__()
        if cfg.d_token % cfg.n_heads or cfg.n_blocks < 1:
            raise ValueError("Invalid attention dimensions")
        self.weight = nn.Parameter(torch.empty(N_FEATURES, cfg.d_token))
        self.bias = nn.Parameter(torch.empty(N_FEATURES, cfg.d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, cfg.d_token))
        for p in (self.weight, self.bias, self.cls):
            nn.init.uniform_(p, -1 / math.sqrt(cfg.d_token), 1 / math.sqrt(cfg.d_token))
        self.blocks = nn.ModuleList([Block(cfg, i == 0) for i in range(cfg.n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_token), nn.ReLU(), nn.Linear(cfg.d_token, len(CLASSES)))

    def forward(self, x):
        tokens = x[..., None] * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, last=i == len(self.blocks) - 1)
        return self.head(tokens[:, 0])


def predict_proba(state, X):
    model, scaler, cfg = state['model'], state['scaler'], state['config']
    device = next(model.parameters()).device
    values = scaler.transform(checked_features(X, allow_empty=True))
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction features')
    model.eval()
    chunks = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                               enabled=cfg.amp and device.type == 'cuda'):
        for start in range(0, len(values), cfg.batch_size):
            x = torch.as_tensor(values[start:start+cfg.batch_size], device=device)
            chunks.append(model(x).float().softmax(1).cpu().numpy())
    result = np.concatenate(chunks) if chunks else np.empty((0, 4), dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite probabilities')
    return result


def fit(train, validation, *, value, seed, settings, sample_weight, epochs=None):
    cfg = FTConfig(**{**settings, 'seed': seed})
    check_table(train, training=True)
    if validation is None and (type(epochs) is not int or epochs < 1):
        raise ValueError('Final refit requires a positive inner-selected epoch count')
    if validation is not None and epochs is not None:
        raise ValueError('Inner fits cannot receive final refit epochs')
    if validation is not None:
        check_table(validation, training=False)
        if set(train.participant_id) & set(validation.participant_id):
            raise ValueError('Participant overlap in training and validation')
    weights = np.asarray(sample_weight, dtype=np.float32)
    if weights.shape != train.y.shape or not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.mean(), 1):
        raise ValueError('Expected aligned positive mean-one sample weights')
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError('Learning rate must be finite and positive')
    device = configure_runtime(cfg, training=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    scaler = fit_standardizer(train.X)
    scaled = scaler.transform(train.X)
    if not np.isfinite(scaled).all():
        raise ValueError('Nonfinite standardized training features')
    X = torch.as_tensor(scaled, device=device)
    y = torch.as_tensor(train.y.astype(np.int64), device=device)
    w = torch.as_tensor(weights, device=device)
    model = FTTransformer(cfg).to(device)
    state = dict(model=model, scaler=scaler, config=cfg)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (decay if param.ndim >= 2 and name not in ('weight', 'bias', 'cls') else no_decay).append(param)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.}], lr=float(value))
    best, best_epoch, best_state, stale = -float('inf'), 0, None, 0
    history = []
    for epoch in range(1, (epochs if validation is None else cfg.max_epochs) + 1):
        model.train()
        start = time.perf_counter()
        order = torch.randperm(len(X), device=device)
        total = 0.
        for offset in range(0, len(X), cfg.batch_size):
            ix = order[offset:offset+cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
                # Weights already have mean one over the whole training fold.
                # Do not re-normalize within each minibatch or apply weights twice.
                loss = (F.cross_entropy(model(X[ix]).float(), y[ix], reduction='none') * w[ix]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            total += loss.item() * len(ix)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        record = {'epoch': epoch, 'loss': total/len(X), 'train_seconds': time.perf_counter()-start}
        if validation is not None:
            start = time.perf_counter()
            proba = predict_proba(state, validation.X)
            score = participant_macro_f1(validation.y, proba.argmax(1), validation.participant_id)
            if not np.isfinite(score):
                raise RuntimeError('Nonfinite participant validation score')
            record.update(validation_macro_f1=score, validation_seconds=time.perf_counter()-start)
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        print(f'epoch={epoch} loss={record["loss"]:.4f} val={record.get("validation_macro_f1", "not_applicable")} train_s={record["train_seconds"]:.2f}', flush=True)
        if validation is not None and stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return state, {'best_epoch': best_epoch if validation is not None else None,
                   'epochs_run': len(history), 'history': history,
                   'device': str(device), 'seed': seed,
                   'epoch_cap_reached': validation is not None and len(history) == cfg.max_epochs,
                   'runtime': runtime_details(device, cfg),
                   'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None}


def serialize(state):
    buffer = io.BytesIO()
    torch.save({'format': 'ft_v2', 'config': asdict(state['config']),
                'classes': list(CLASSES), 'feature_count': N_FEATURES,
                'mean': torch.from_numpy(state['scaler'].mean.copy()),
                'scale': torch.from_numpy(state['scaler'].scale.copy()),
                'state_dict': {k: v.detach().cpu() for k, v in state['model'].state_dict().items()}}, buffer)
    return buffer.getvalue()


def deserialize(payload):
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if (saved.get('format') != 'ft_v2' or saved.get('classes') != list(CLASSES)
            or saved.get('feature_count') != N_FEATURES):
        raise ValueError('Unknown or incompatible FT checkpoint; pilot checkpoints cannot resume a formal run')
    cfg = FTConfig(**saved['config'])
    mean, scale = saved['mean'].numpy(), saved['scale'].numpy()
    if (mean.shape != (N_FEATURES,) or scale.shape != (N_FEATURES,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('Invalid FT checkpoint standardizer')
    device = configure_runtime(cfg, training=False)
    model = FTTransformer(cfg).to(device)
    model.load_state_dict(saved['state_dict'])
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite FT checkpoint parameters')
    model.eval()
    return dict(model=model, config=cfg, scaler=Standardizer(mean, scale))
