"""Numerical FT adapter; compact architecture from the project benchmark.
Reference: https://github.com/yandex-research/rtdl-revisiting-models
"""
import io, math, random, time
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data import Standardizer, fit_standardizer
from metrics import participant_macro_f1

@dataclass(frozen=True)
class FTConfig:
    d_token: int = 64
    n_blocks: int = 2
    n_heads: int = 4
    ffn_hidden: int = 128
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 8
    weight_decay: float = 1e-5
    seed: int = 17
    amp: bool = True


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
        self.weight = nn.Parameter(torch.empty(135, cfg.d_token))
        self.bias = nn.Parameter(torch.empty(135, cfg.d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, cfg.d_token))
        for p in (self.weight, self.bias, self.cls):
            nn.init.uniform_(p, -1 / math.sqrt(cfg.d_token), 1 / math.sqrt(cfg.d_token))
        self.blocks = nn.ModuleList([Block(cfg, i == 0) for i in range(cfg.n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_token), nn.ReLU(), nn.Linear(cfg.d_token, 4))

    def forward(self, x):
        tokens = x[..., None] * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, last=i == len(self.blocks) - 1)
        return self.head(tokens[:, 0])


def predict_proba(state, X):
    model, scaler, cfg = state['model'], state['scaler'], state['config']
    device = next(model.parameters()).device
    values = scaler.transform(X)
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
    if validation is None and (type(epochs) is not int or epochs < 1):
        raise ValueError('Final refit requires a positive inner-selected epoch count')
    if validation is not None and epochs is not None:
        raise ValueError('Inner fits cannot receive final refit epochs')
    if validation is not None and set(train.participant_id) & set(validation.participant_id):
        raise ValueError('Participant overlap in training and validation')
    if train.X.ndim != 2 or train.X.shape[1] != 135 or not np.array_equal(np.unique(train.y), np.arange(4)):
        raise ValueError('Expected 135 features and four training classes')
    weights = np.asarray(sample_weight, dtype=np.float32)
    if weights.shape != train.y.shape or not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.mean(), 1):
        raise ValueError('Expected aligned positive mean-one sample weights')
    if not np.isfinite(value) or value <= 0 or min(cfg.batch_size, cfg.max_epochs, cfg.patience) < 1:
        raise ValueError('Invalid learning rate or training settings')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.amp and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('amp=True requires a CUDA GPU supporting BF16')
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    scaler = fit_standardizer(train.X)
    X = torch.as_tensor(scaler.transform(train.X), device=device)
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
                   'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None}


def serialize(state):
    buffer = io.BytesIO()
    torch.save({'format': 'ft_v1', 'config': asdict(state['config']),
                'mean': torch.from_numpy(state['scaler'].mean.copy()),
                'scale': torch.from_numpy(state['scaler'].scale.copy()),
                'state_dict': {k: v.detach().cpu() for k, v in state['model'].state_dict().items()}}, buffer)
    return buffer.getvalue()


def deserialize(payload):
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if saved['format'] != 'ft_v1':
        raise ValueError('Unknown FT checkpoint format')
    cfg = FTConfig(**saved['config'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FTTransformer(cfg).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return dict(model=model, config=cfg, scaler=Standardizer(saved['mean'].numpy(), saved['scale'].numpy()))
