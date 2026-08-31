#!/usr/bin/env python3
"""A model whose unit is the SHOT, not the frame: one pass forward through time and one back.

Everything else in the zoo scores each frame on its own 122 numbers. That is a real limitation
rather than a stylistic one — the vessel's eddy currents and the plasma's current profile are
STATE, evolving on their own timescales, and no set of instantaneous magnetics can express a
quantity whose value depends on what the shot was doing 200 ms ago. A memoryless model can only
approximate that state by whatever of it happens to be visible now.

**The sequence branch starts at exactly zero.** `delta` is a linear layer initialised to zeros, so
at step 0 this model's output IS the per-frame MLP's, to the last bit. Whatever the recurrence
contributes it has to earn against that baseline during training, and the comparison against the
MLP is therefore about the recurrence and not about a differently-initialised net.

**It trains at `frame_share = 1.0` and refuses less.** Training on a fifth of the frames means a
100 ms step where inference sees 20 ms, so the model would learn dynamics at one spacing and be
asked for them at another. Measured on 202 shots: 97.9% of steps are 20 ms and every step is a
multiple of it — but 77% of shots drop at least one frame, the largest gap is 2580 ms, and shots
run from 2 to 373 frames. Even at 1.0 the grid is irregular, which is why the gap between frames
has to be an INPUT (`features.frame_gaps`), in units of the base step so that a hole reads as
"this step was three frames long".

Inference needs nothing new: `predict_row` already hands a model one shot's frames in time order,
so `predict` takes its argument as one sequence — see the guard in `predict`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from my_experiments.models import (
    ACTIVATIONS,
    LR_SCHEDULES,
    FloatArray,
    TargetModel,
    build_mlp,
    resolve_device,
)
from toolkit.progress import bar_kwargs

# A shot is at most 373 frames on this corpus. `predict` is documented to take its input as ONE
# shot, and the one way to break that quietly is to hand it several concatenated — which would
# still return the right SHAPE, computed from a sequence that never existed. Anything past this
# is not a shot.
MAX_SHOT_FRAMES = 1000

# How often the fit writes a train/val line, in optimizer steps. A logging cadence, not a
# hyper-parameter; see models.MLP_LOG_EVERY_STEPS.
SEQ_LOG_EVERY_STEPS = 500


def _pad_index(starts: np.ndarray, lengths: np.ndarray,
               ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row indices and a validity mask for a padded (B, T_max) batch of the flat frame array.

    The frames of every shot are contiguous in the concatenated arrays, so a batch is arithmetic
    rather than a gather: `starts[i] + arange(T_max)`, clipped at each shot's own length. The
    clipped positions are masked, and what they point AT never matters — they are dropped from the
    loss and the GRU is packed to its true lengths.
    """
    lens = lengths[ids]
    t_max = int(lens.max())
    offsets = np.arange(t_max)[None, :]
    mask = offsets < lens[:, None]
    rows = starts[ids][:, None] + np.minimum(offsets, lens[:, None] - 1)
    return rows, mask


# What reads the shot. The field is still called `seq_cell` and the attribute is still `self.gru`,
# because renaming either would change the state-dict keys and every `seq` member already sitting in
# `artefacts/lanes/` would stop loading — a rename that breaks existing blends to improve a name.
#
# `gru`/`lstm` are recurrent; `conv` and `attn` are not, and that is the point of having them. See
# single_model_hypotheses.md, "The shot-mixer family": the criterion for this model is DISAGREEMENT
# with the MLPs, not accuracy, and a deeper GRU errs where a shallow one errs.
SEQ_CELLS = ("gru", "lstm", "conv", "attn")
RECURRENT_CELLS = ("gru", "lstm")


def _activation(name: str) -> Any:
    import torch
    acts = {"relu": torch.nn.ReLU, "gelu": torch.nn.GELU, "silu": torch.nn.SiLU,
            "tanh": torch.nn.Tanh}
    if name not in acts:
        raise ValueError(f"torch_seq: unknown activation {name!r}; known {sorted(acts)}")
    return acts[name]


def _dilated_stack(width: int, out_width: int, dilations: list[int], activation: str,
                   dropout: float) -> Any:
    """S8. Dilated temporal convolutions over the shot, masked at every layer.

    A shot is ~220 frames at 20 ms and a GRU carries information across the ramp through 220
    SEQUENTIAL steps. Dilations 1,2,4,8,16,32 reach a 127-frame receptive field in six layers, in
    parallel, and structure their errors by LOCALITY where recurrence structures them by recency.

    Symmetric padding, so it reads both directions for the same reason the GRU is bidirectional: a
    submission is scored offline over complete shots, with no causality constraint to respect.

    Residual blocks, so at initialisation the stack is close to the identity on `enc`'s output and
    the correction still has to earn what it does — the same discipline as `delta` starting at zero.
    """
    import torch

    act = _activation(activation)

    class Dilated(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                torch.nn.Sequential(
                    torch.nn.Conv1d(width, width, 3, padding=d, dilation=d),
                    act(), torch.nn.Dropout(dropout))
                for d in dilations)
            self.out = torch.nn.Conv1d(width, out_width, 1)

        def forward(self, h: Any, mask: Any) -> Any:
            # (B, T, C) -> (B, C, T), which is what Conv1d reads.
            x = h.transpose(1, 2)
            m = mask[:, None, :].to(x.dtype)
            # THE MASK IS REAPPLIED BEFORE EVERY LAYER, not once at the end. A convolution has no
            # notion of a packed sequence: with the padded tail left in, layer 1 spreads it one
            # frame into the real data, layer 2 two more, and by layer 6 the pad value has reached
            # 63 frames inside every short shot. It does not crash, does not change any shape, and
            # trains the model on the padding.
            x = x * m
            for block in self.blocks:
                x = (x + block(x)) * m
            return self.out(x).transpose(1, 2)

    return Dilated()


def _shot_attention(width: int, out_width: int, layers: int, heads: int, activation: str,
                    dropout: float) -> Any:
    """S9. A small transformer encoder over the shot.

    Bidirectional by construction, so reading the future needs no separate justification, and at
    T ~ 220 the attention matrix is 48k entries — free. Furthest of the three mixers from a GRU in
    HOW it errs: attention is content-addressed where recurrence is position-addressed, so the
    frames it confuses are chosen differently. That is the property being bought.

    Sinusoidal positions rather than learned ones: shots differ in length and a learned table would
    have to be sized for the longest, leaving its tail trained on almost nothing.
    """
    import torch

    class Attn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer = torch.nn.TransformerEncoderLayer(
                d_model=width, nhead=heads, dim_feedforward=4 * width, dropout=dropout,
                activation=activation if activation in ("relu", "gelu") else "gelu",
                batch_first=True, norm_first=True)
            self.tr = torch.nn.TransformerEncoder(layer, layers)
            self.out = torch.nn.Linear(width, out_width)
            pos = torch.zeros(4096, width)
            t = torch.arange(4096, dtype=torch.float32)[:, None]
            f = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                          * (-np.log(10000.0) / width))[None, :]
            pos[:, 0::2] = torch.sin(t * f)
            pos[:, 1::2] = torch.cos(t * f)
            self.register_buffer("pos", pos, persistent=False)

        def forward(self, h: Any, mask: Any) -> Any:
            x = h + self.pos[:h.shape[1]][None]
            # `src_key_padding_mask` marks what to IGNORE, so it is the mask inverted. No row is
            # fully padded — every shot has at least one frame — which is what keeps the softmax
            # from meeting an all-masked row and returning NaN.
            y = self.tr(x, src_key_padding_mask=~mask)
            return self.out(y) * mask[:, :, None].to(y.dtype)

    return Attn()


@dataclass
class TorchSeqModel(TargetModel):
    """Per-frame encoder, a bidirectional GRU over the shot, and a zero-initialised residual head.

    The three pieces, and why each is there:

    - `head` is the per-frame MLP, unchanged — same `hidden_sizes`, same activation. It carries the
      instantaneous part of the mapping, which is most of it: the MLP alone reaches 0.99.
    - `enc` + `gru` read the whole shot. Bidirectional because a submission is scored offline over
      complete shots, so a frame may legitimately be informed by what happened AFTER it; there is
      no causality constraint to respect here, and half the available context would be thrown away
      by pretending otherwise.
    - `delta` maps the recurrent state to a correction of the head's output, and starts at zero.

    Like the MLP it expects targets that TargetScaler has already scaled, keeps its weights as
    numpy arrays so the artifact unpickles without torch, and predicts on the CPU whatever
    `device` the fit used.
    """

    hidden_sizes: list[int]
    # Width of the recurrent state, per direction, and how many GRU layers are stacked. The two
    # directions are concatenated before `delta`, so it sees 2 * seq_hidden.
    seq_hidden: int
    seq_layers: int
    # Width of a SECOND recurrent stage on top of the first, per direction, or 0 to stack inside
    # one module via `seq_layers`. The two are not the same knob and the difference is exactly what
    # this field exists for: torch's stacked GRU gives every layer the same hidden size, so
    # `seq_layers: 2` can only ever ask for 256 -> 256. A second module can be NARROWER, which is
    # the shape worth asking about — the first stage sees 84 features per frame and has a whole
    # shot to summarise, the second sees an already-summarised 512 and only has to refine it.
    #
    # At `seq_hidden2 == seq_hidden` this is `seq_layers: 2` in all but name (torch's second layer
    # also takes 2 * hidden as its input, being bidirectional), which makes the pair a consistency
    # check on each other rather than two independent guesses.
    seq_hidden2: int
    # S10. Feed `enc`'s per-frame output to `delta` ALONGSIDE the shot-level state, instead of only
    # the latter. Without it the correction sees the recurrence and nothing else, so anything
    # per-frame it needs has to be re-derived inside the recurrence from a representation `enc` has
    # already computed. With it, the recurrence is responsible for what is genuinely temporal and
    # nothing more. It cannot cost anything at initialisation — `delta` still starts at zero.
    seq_skip: bool
    # `seq_cell: conv` only. Dilations of the temporal stack, widest last; the receptive field is
    # 2 * sum(dilations) + 1 frames, so 1,2,4,8,16,32 reads 127 frames of the 20 ms series either
    # side. Depth IS this list's length.
    seq_conv_dilations: str
    # `seq_cell: attn` only. Attention heads; `seq_layers` is the number of encoder layers.
    seq_attn_heads: int
    # Which recurrent cell. `gru` is every number before 2026-08-30. `lstm` exists for DIVERSITY,
    # not for accuracy: `seq` alone loses to the MLP, 0.9914 against 0.9931, and earns its place
    # only in the average — where `mlp+seq` = 0.9943 beats `mlp` plus three extra seeds at 0.9940,
    # so one extra ARCHITECTURE beats three extra seeds. The question for this field is therefore
    # not whether the LSTM scores higher but whether it errs ELSEWHERE: seeds correlate 0.56 and
    # feature sets 0.75, and a cell that lands below 0.56 is a better diversity axis than a seed.
    #
    # The two are drop-in: both take (input, hidden, layers) and return (output, state), and the
    # state is discarded here, so only the constructor differs.
    seq_cell: str
    # Width of a hidden layer between the recurrent state and the correction, or 0 for the single
    # linear map every number before 2026-08-31 was measured with. The recurrence produces a
    # 2 * seq_hidden state and `delta` has to turn it into 52 target coordinates; with one linear
    # layer the ONLY nonlinearity the sequence branch owns lives in `enc`, before the recurrence,
    # so whatever the GRU represents can reach the targets in a linear combination and no other
    # way. `head` — the per-frame trunk this corrects — is two hidden layers deep for the same
    # 52 outputs, which is the asymmetry this field exists to test.
    #
    # NARROWER than 2 * seq_hidden is the interesting setting rather than an economy: the
    # correction is small by construction (`delta` starts at zero and only grows what it earns), so
    # a bottleneck says the shot-level information is low-dimensional, which is a claim about the
    # problem and not about the parameter count.
    seq_head_hidden: int
    # Shots per optimizer step. The unit of a batch is the SHOT here, so this is not comparable to
    # the MLP's `batch_size` in rows: 32 shots is about 7500 frames at full frames.
    batch_shots: int
    max_steps: int
    patience_steps: int
    eval_every_steps: int
    learning_rate: float
    weight_decay: float
    seed: int
    threads: int
    device: str
    activation: str
    dropout: float
    lr_schedule: str
    lr_final_factor: float
    lr_t_max_steps: int
    # `mse`, or `huber` with the transition at `huber_delta` standard deviations of the scaled
    # target. SCALED BY 2 against torch's convention, which is 0.5*x^2 in the quadratic region
    # where MSELoss is x^2: without the factor a huber arm would also halve every ordinary
    # gradient, and `grad_clip` is scale-sensitive even though Adam is not — so the arm would
    # differ from its control in how often the clip fires as well as in outlier handling. The
    # factor makes the two losses agree wherever no clipping happens, which is the point of a
    # controlled arm.
    loss: str
    huber_delta: float
    # Decay of an exponential moving average of the WEIGHTS, judged beside the live ones at every
    # evaluation; the fit keeps whichever validates better, so it can only help or tie. 0.0 is off.
    ema_decay: float
    _state: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _ema_wins: int = field(default=0, init=False, repr=False)
    _n_in: int = field(default=0, init=False, repr=False)
    _n_out: int = field(default=0, init=False, repr=False)
    _best_step: int = field(default=-1, init=False, repr=False)
    _ran_steps: int = field(default=0, init=False, repr=False)
    _delta_share: float = field(default=float("nan"), init=False, repr=False)

    @property
    def kind(self) -> str:
        return "torch_seq"

    def _build(self, n_in: int, n_out: int) -> Any:
        import torch

        if self.activation not in ACTIVATIONS:
            raise ValueError(f"torch_seq: unknown activation {self.activation!r}; "
                             f"known {sorted(ACTIVATIONS)}")

        class Net(torch.nn.Module):
            def __init__(self, cfg: TorchSeqModel) -> None:
                super().__init__()
                self.head = build_mlp(n_in, cfg.hidden_sizes, n_out, cfg.activation, cfg.dropout)
                self.enc = build_mlp(n_in, cfg.hidden_sizes, cfg.seq_hidden, cfg.activation,
                                     cfg.dropout)
                if cfg.seq_cell not in SEQ_CELLS:
                    raise ValueError(f"torch_seq: seq_cell {cfg.seq_cell!r}, known {SEQ_CELLS}")
                # `self.packed` says whether the mixer consumes a PackedSequence. A plain bool, so
                # it stays out of the state dict.
                self.packed = cfg.seq_cell in RECURRENT_CELLS
                self.gru2 = None
                if self.packed:
                    cell = torch.nn.GRU if cfg.seq_cell == "gru" else torch.nn.LSTM
                    self.gru = cell(cfg.seq_hidden, cfg.seq_hidden, cfg.seq_layers,
                                    batch_first=True, bidirectional=True,
                                    dropout=cfg.dropout if cfg.seq_layers > 1 else 0.0)
                    # A second stage, of its own width. Both directions of the first are
                    # concatenated into its input, which is why it reads 2 * seq_hidden.
                    self.gru2 = (cell(2 * cfg.seq_hidden, cfg.seq_hidden2, 1, batch_first=True,
                                      bidirectional=True) if cfg.seq_hidden2 else None)
                    width = 2 * (cfg.seq_hidden2 or cfg.seq_hidden)
                else:
                    # Both non-recurrent mixers emit 2 * seq_hidden, so `delta` faces the same
                    # width whichever reads the shot and the arms differ in one thing only.
                    width = 2 * cfg.seq_hidden
                    if cfg.seq_cell == "conv":
                        dil = [int(d) for d in cfg.seq_conv_dilations.split(",") if d.strip()]
                        if not dil or any(d < 1 for d in dil):
                            raise ValueError(f"torch_seq: seq_conv_dilations "
                                             f"{cfg.seq_conv_dilations!r} is not a list of "
                                             f"positive integers")
                        self.gru = _dilated_stack(cfg.seq_hidden, width, dil, cfg.activation,
                                                  cfg.dropout)
                    else:
                        if cfg.seq_hidden % cfg.seq_attn_heads:
                            raise ValueError(f"torch_seq: seq_attn_heads {cfg.seq_attn_heads} does "
                                             f"not divide seq_hidden {cfg.seq_hidden}")
                        self.gru = _shot_attention(cfg.seq_hidden, width, cfg.seq_layers,
                                                   cfg.seq_attn_heads, cfg.activation, cfg.dropout)
                # S10: the correction also sees `enc`'s per-frame output, which is seq_hidden wide.
                self.skip = bool(cfg.seq_skip)
                if self.skip:
                    width = width + cfg.seq_hidden
                # One linear map, or a hidden layer first — built by `build_mlp` either way, so
                # the activation and dropout conventions stay declared in one place.
                self.delta = (build_mlp(width, [cfg.seq_head_hidden], n_out,
                                        cfg.activation, cfg.dropout) if cfg.seq_head_hidden
                              else torch.nn.Linear(width, n_out))
                # The whole point of the residual form: at initialisation this model IS `head`.
                # Zeroing the OUTPUT layer, found rather than indexed — `build_mlp` may put
                # dropout or a norm after the last Linear and an index would silently zero those.
                last = (self.delta if isinstance(self.delta, torch.nn.Linear) else
                        [m for m in self.delta.modules() if isinstance(m, torch.nn.Linear)][-1])
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)

            def parts(self, x: Any, lengths: Any) -> tuple[Any, Any]:
                """The per-frame prediction and the correction the shot adds to it, separately.

                Kept apart because their RELATIVE size is the diagnostic this architecture exists
                to produce: `delta` starts at zero, so how big it grows is how much the model
                decided the rest of the shot was worth. A score that matches the MLP with a
                correction near zero and a score that matches it with a large one are different
                results, and the composite cannot tell them apart.
                """
                h = self.enc(x)
                if self.packed:
                    # `lengths` is on the CPU, as pack_padded_sequence requires, and
                    # enforce_sorted=False lets the batch keep its natural order.
                    packed = torch.nn.utils.rnn.pack_padded_sequence(
                        h, lengths, batch_first=True, enforce_sorted=False)
                    out, _ = self.gru(packed)
                    if self.gru2 is not None:
                        # Straight through, still packed: unpacking between the stages would feed
                        # the second one the padding the first was carefully never shown.
                        out, _ = self.gru2(out)
                    seq, _ = torch.nn.utils.rnn.pad_packed_sequence(
                        out, batch_first=True, total_length=x.shape[1])
                else:
                    # The same guarantee packing gives the recurrent path, written out: which
                    # frames are real. `lengths` is on the CPU by the packing convention above, so
                    # it is moved here rather than assumed to be beside x.
                    mask = (torch.arange(x.shape[1], device=x.device)[None, :]
                            < lengths.to(x.device)[:, None])
                    seq = self.gru(h, mask)
                return self.head(x), self.delta(
                    torch.cat([seq, h], dim=-1) if self.skip else seq)

            def forward(self, x: Any, lengths: Any) -> Any:
                # (B, T, n_in) -> (B, T, n_out).
                per_frame, correction = self.parts(x, lengths)
                return per_frame + correction

        return Net(self)

    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray,
            shots: np.ndarray | None = None, shots_val: np.ndarray | None = None,
            monitor: Callable[[FloatArray], float] | None = None,
            monitor_idx: np.ndarray | None = None,
            head_y: FloatArray | None = None,
            head_y_val: FloatArray | None = None,
            head_n_pca: int = 0) -> None:
        self._check_fit_input(X, Y, X_val, Y_val)
        import torch

        if shots is None or shots_val is None:
            raise ValueError(
                "torch_seq: needs the per-shot frame counts, and got frames with no boundaries. "
                "The reader returns them; a caller that concatenates and drops them is asking "
                "this model to read one shot's future off the next shot's past."
            )
        for name, lens, n in (("train", shots, len(X)), ("val", shots_val, len(X_val))):
            if int(np.sum(lens)) != n:
                raise ValueError(f"torch_seq: {name} shot lengths sum to {int(np.sum(lens))}, "
                                 f"against {n} rows")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"torch_seq: unknown lr_schedule {self.lr_schedule!r}; "
                             f"known {sorted(LR_SCHEDULES)}")
        for name, value in (("max_steps", self.max_steps),
                            ("eval_every_steps", self.eval_every_steps),
                            ("batch_shots", self.batch_shots)):
            if value < 1:
                raise ValueError(f"torch_seq: {name} is {value}, must be >= 1")

        dev = resolve_device(self.device, "torch_seq")
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        self._n_in, self._n_out = X.shape[1], Y.shape[1]
        net = self._build(self._n_in, self._n_out).to(dev)

        starts = np.concatenate([[0], np.cumsum(shots)[:-1]]).astype(np.int64)
        starts_val = np.concatenate([[0], np.cumsum(shots_val)[:-1]]).astype(np.int64)
        lengths, lengths_val = np.asarray(shots, dtype=np.int64), np.asarray(shots_val, np.int64)

        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
        Yt = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(dev)
        Xvt = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)).to(dev)
        Yvt = torch.from_numpy(np.ascontiguousarray(Y_val, dtype=np.float32)).to(dev)

        def batch_loss(Xs: Any, Ys: Any, starts_: np.ndarray, lengths_: np.ndarray,
                       ids: np.ndarray) -> tuple[Any, int]:
            """Summed squared error over the VALID frames of one batch, and how many there were.

            Summed and divided by the true frame count rather than averaged per shot: a mean over
            shots would weight a 42-frame shot as heavily as a 373-frame one, which is a different
            loss from the one every other model in the zoo is fitted on.
            """
            rows, mask = _pad_index(starts_, lengths_, ids)
            idx = torch.from_numpy(rows).to(dev)
            keep = torch.from_numpy(mask).to(dev)
            xb, yb = Xs[idx], Ys[idx]
            pred = net(xb, torch.from_numpy(lengths_[ids]))
            if self.loss == "mse":
                el = (pred - yb) ** 2
            else:
                el = 2.0 * torch.nn.functional.huber_loss(
                    pred, yb, reduction="none", delta=self.huber_delta)
            err = el.mean(dim=2) * keep
            return err.sum(), int(mask.sum())

        if self.loss not in ("mse", "huber"):
            raise ValueError(f"torch_seq: unknown loss {self.loss!r}; known ['huber', 'mse']")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"torch_seq: ema_decay is {self.ema_decay}, must be in [0, 1)")
        # Cloned, not referenced: the average must not follow `opt.step()`.
        ema = ({k: v.detach().clone() for k, v in net.state_dict().items()}
               if self.ema_decay else None)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                               weight_decay=self.weight_decay, fused=(dev == "cuda"))
        sched = None
        t_max = self.lr_t_max_steps or self.max_steps
        if self.lr_schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=t_max, eta_min=self.learning_rate * self.lr_final_factor)

        rng = np.random.default_rng(self.seed)
        order, cursor = rng.permutation(len(lengths)), 0

        def next_batch() -> np.ndarray:
            """The next `batch_shots` shot ids, reshuffling when the list runs out."""
            nonlocal order, cursor
            if cursor + self.batch_shots > len(order):
                order, cursor = rng.permutation(len(lengths)), 0
            ids = order[cursor:cursor + self.batch_shots]
            cursor += self.batch_shots
            return ids

        best_loss, best_state, since_best = float("inf"), None, 0
        run_t, run_n = torch.zeros((), device=dev), 0
        bar = tqdm(total=self.max_steps, desc="    seq", unit="step",
                   **bar_kwargs(off_in_log=True))
        for step in range(1, self.max_steps + 1):
            opt.zero_grad()
            total, n = batch_loss(Xt, Yt, starts, lengths, next_batch())
            (total / n).backward()
            opt.step()
            run_t += total.detach()
            run_n += n
            if ema is not None:
                with torch.no_grad():
                    for k, v in net.state_dict().items():
                        # Integer buffers cannot take a decayed average; they are counters, so the
                        # live value is the only meaningful one.
                        if ema[k].is_floating_point():
                            ema[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)
                        else:
                            ema[k].copy_(v)
            if sched is not None and step <= t_max:
                sched.step()

            self._ran_steps = step
            last = step >= self.max_steps
            if step % self.eval_every_steps and not last:
                continue

            net.eval()
            with torch.no_grad():
                def measure() -> float:
                    v_tot, v_n = torch.zeros((), device=dev), 0
                    for i in range(0, len(lengths_val), self.batch_shots):
                        ids = np.arange(i, min(i + self.batch_shots, len(lengths_val)))
                        tt, kk = batch_loss(Xvt, Yvt, starts_val, lengths_val, ids)
                        v_tot += tt
                        v_n += kk
                    return float(v_tot) / v_n

                val_loss = measure()
                # A20 for the recurrence: the running weight average is a second candidate, judged
                # by the same rule at the same moment, so this cannot lose to not having it.
                from_ema = False
                if ema is not None:
                    live = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    net.load_state_dict(ema)
                    ema_loss = measure()
                    if ema_loss < val_loss:
                        val_loss, from_ema = ema_loss, True
                        self._ema_wins += 1
                    net.load_state_dict(live)
            net.train()
            train_loss = float(run_t) / run_n
            run_t, run_n = torch.zeros((), device=dev), 0

            if val_loss < best_loss or not self.patience_steps:
                best_loss, since_best, self._best_step = val_loss, 0, step
                keep = ema if (from_ema and ema is not None) else net.state_dict()
                best_state = {k: v.detach().cpu().numpy().copy() for k, v in keep.items()}
            else:
                since_best += self.eval_every_steps
            bar.update(step - bar.n)
            bar.set_postfix(mse=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                            best=f"{best_loss:.4f}@{self._best_step}", refresh=False)
            head = (f"    seq step {step:7d}: train {train_loss:.6f}  val {val_loss:.6f}"
                    f"  (MultiRMSE {np.sqrt(self._n_out * val_loss):.4f})")
            if step % SEQ_LOG_EVERY_STEPS < self.eval_every_steps:
                bar.write(f"{head}  best {best_loss:.6f} @ {self._best_step}")
            if last or (self.patience_steps and since_best >= self.patience_steps):
                if not last:
                    bar.write(f"{head}  -> stopping, {since_best} steps since best "
                              f"{best_loss:.6f} @ {self._best_step}")
                break
        if best_state is None:
            raise RuntimeError("torch_seq: no evaluation completed — the fit ran no steps")
        self._state = best_state

        # How much of the answer came from the shot rather than from the frame, measured on the
        # weights that were actually kept. Reported whatever it says: a model that scores well with
        # a correction of 1% of the per-frame prediction has NOT shown that reading the shot helps,
        # and one that scores the same with a correction of 40% has found something real and spent
        # it badly. The score cannot distinguish those two.
        net.load_state_dict({k: torch.from_numpy(v) for k, v in best_state.items()})
        net.eval()
        with torch.no_grad():
            ids = np.arange(min(self.batch_shots, len(lengths_val)))
            rows, mask = _pad_index(starts_val, lengths_val, ids)
            keep = torch.from_numpy(mask).to(dev)[:, :, None]
            frame, corr = net.parts(Xvt[torch.from_numpy(rows).to(dev)],
                                    torch.from_numpy(lengths_val[ids]))
            rms = [float(torch.sqrt(((p * keep) ** 2).sum() / keep.sum() / p.shape[2]))
                   for p in (frame, corr)]
        self._delta_share = rms[1] / rms[0] if rms[0] > 0 else float("nan")

    def fit_report(self) -> str:
        if self._best_step < 0:
            return ""
        if not self.patience_steps:
            where = f", early stopping off, kept step {self._best_step}"
        else:
            ceiling = f" (of {self.max_steps} allowed)" if self._ran_steps < self.max_steps else ""
            where = f", best step {self._best_step} of {self._ran_steps} run{ceiling}"
        # Reported even when it is zero, and that is the point: a run whose averaged weights won
        # NOTHING is indistinguishable in its score from a run where the averaging was never wired
        # in at all. Printing the count is what separates "measured and lost" from a silent no-op.
        ema = (f", averaged weights won {self._ema_wins} evaluation(s)" if self.ema_decay else "")
        return (f"{where}, the shot moved the prediction by {self._delta_share:.1%} of it{ema}")

    def predict(self, X: FloatArray) -> FloatArray:
        """(T, n_targets) for ONE shot, whose frames are `X` in time order.

        That is what `predict_row` passes and what the metric scores; a concatenation of several
        shots would return the right shape from a sequence that never happened, so the length is
        checked rather than trusted.
        """
        if self._state is None:
            raise RuntimeError("torch_seq: predict() before fit()")
        if len(X) > MAX_SHOT_FRAMES:
            raise ValueError(
                f"torch_seq: predict() got {len(X)} frames, and takes ONE shot at a time — the "
                f"longest shot in this corpus is 373 frames. Several shots concatenated would be "
                f"read as one sequence, silently."
            )
        import torch

        torch.set_num_threads(self.threads)
        net = self._build(self._n_in, self._n_out)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in self._state.items()})
        net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))[None, :, :]
            out = net(xb, torch.tensor([len(X)]))[0].numpy().astype(np.float64)
        return self._check_predict_output(X, out)
