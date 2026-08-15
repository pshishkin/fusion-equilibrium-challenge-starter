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
from my_experiments.progress import bar_kwargs

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
    _state: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
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
                self.gru = torch.nn.GRU(cfg.seq_hidden, cfg.seq_hidden, cfg.seq_layers,
                                        batch_first=True, bidirectional=True,
                                        dropout=cfg.dropout if cfg.seq_layers > 1 else 0.0)
                self.delta = torch.nn.Linear(2 * cfg.seq_hidden, n_out)
                # The whole point of the residual form: at initialisation this model IS `head`.
                torch.nn.init.zeros_(self.delta.weight)
                torch.nn.init.zeros_(self.delta.bias)

            def parts(self, x: Any, lengths: Any) -> tuple[Any, Any]:
                """The per-frame prediction and the correction the shot adds to it, separately.

                Kept apart because their RELATIVE size is the diagnostic this architecture exists
                to produce: `delta` starts at zero, so how big it grows is how much the model
                decided the rest of the shot was worth. A score that matches the MLP with a
                correction near zero and a score that matches it with a large one are different
                results, and the composite cannot tell them apart.
                """
                # `lengths` is on the CPU, as pack_padded_sequence requires, and
                # enforce_sorted=False lets the batch keep its natural order.
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    self.enc(x), lengths, batch_first=True, enforce_sorted=False)
                out, _ = self.gru(packed)
                seq, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    out, batch_first=True, total_length=x.shape[1])
                return self.head(x), self.delta(seq)

            def forward(self, x: Any, lengths: Any) -> Any:
                # (B, T, n_in) -> (B, T, n_out).
                per_frame, correction = self.parts(x, lengths)
                return per_frame + correction

        return Net(self)

    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray,
            shots: np.ndarray | None = None, shots_val: np.ndarray | None = None,
            monitor: Callable[[FloatArray], float] | None = None,
            monitor_idx: np.ndarray | None = None) -> None:
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
            err = ((pred - yb) ** 2).mean(dim=2) * keep
            return err.sum(), int(mask.sum())

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
            if sched is not None and step <= t_max:
                sched.step()

            self._ran_steps = step
            last = step >= self.max_steps
            if step % self.eval_every_steps and not last:
                continue

            net.eval()
            with torch.no_grad():
                v_tot, v_n = torch.zeros((), device=dev), 0
                for i in range(0, len(lengths_val), self.batch_shots):
                    ids = np.arange(i, min(i + self.batch_shots, len(lengths_val)))
                    t, k = batch_loss(Xvt, Yvt, starts_val, lengths_val, ids)
                    v_tot += t
                    v_n += k
                val_loss = float(v_tot) / v_n
            net.train()
            train_loss = float(run_t) / run_n
            run_t, run_n = torch.zeros((), device=dev), 0

            if val_loss < best_loss or not self.patience_steps:
                best_loss, since_best, self._best_step = val_loss, 0, step
                best_state = {k: v.detach().cpu().numpy().copy()
                              for k, v in net.state_dict().items()}
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
        return f"{where}, the shot moved the prediction by {self._delta_share:.1%} of it"

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
