import warnings

from lightning.pytorch import LightningModule
from typing import Literal, List, Mapping, Any

from torch import Tensor

from torch.nn.attention.flex_attention import create_block_mask

from ptp.attention import (
    make_ar_mask_mod,
    make_completion_mask_mod,
)
from ptp.data.collate import IGNORE_INDEX
from ptp.data.utils import predict_bin_edges
from ptp.transformer import TransformerModel, MixedTransformerModel
import torch
import numpy as np
from transformers.cache_utils import DynamicCache, StaticCache

from ptp.utils import instantiate


class ParallelSamplingLightningModule(LightningModule):
    def __init__(self, optim_cfg: dict = None,
                 model_cfg: Mapping[str, Any] | None = None, model: MixedTransformerModel | None = None,
                 completion_loss_weight: float = 1.0,
                 completion_gamma: float = 1.0,
                 pbar_metrics: List[str] | None = None,
                 tokens_per_student_call: int = 20,
                 temperature: float | None = None,
                 top_k: int | None = None,
                 top_p: float | None = None,
                 hist_base: list[float] | None = None):
        if pbar_metrics is None:
            pbar_metrics = ['correct']
            if completion_loss_weight > 0.0:
                pbar_metrics.append('l_completion')
        super().__init__()
        if (model is None) == (model_cfg is None):
            raise ValueError("Exactly one of model and model_cfg must be provided, got "
                             f"model: {model is None=}, model_cfg: {model_cfg is None=}")
        self.model_cfg = model_cfg
        self.model: MixedTransformerModel | None = model

        self.optim_cfg = optim_cfg
        self.completion_loss_weight = completion_loss_weight
        self.completion_gamma = completion_gamma

        self.pbar_metrics = pbar_metrics

        self.tokens_per_student_call = tokens_per_student_call
        self.total_token_budget: int | None = None
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.hist_base = torch.tensor(hist_base, dtype=torch.float64) if hist_base is not None else None
        self._hist_accumulator: list[torch.Tensor] = []

    def configure_model(self) -> None:
        if self.model is None:
            self.model = MixedTransformerModel(**self.model_cfg)

    def enter_inference_mode(self, gate_window: int):
        """Switch to inference mode: fuse LoRA weights into GatedLinearLoraMerged."""
        self.model.enter_inference_mode(gate_window)

    def exit_inference_mode(self):
        """Switch back to training mode: restore GatedLinearLora layers."""
        self.model.exit_inference_mode()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Rename any keys starting with "student." to "model."
        renamed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("student."):
                key = key.replace("student.", "model.", 1)
            if ".u_adapter." in key:
                key = key.replace(".u_adapter.", ".u_embed.", 1)
            renamed_state_dict[key] = value

        # Adapter-only checkpoints intentionally omit base-model parameters.
        # Auto-relax strict loading for that case while preserving strict behavior
        # for full checkpoints.
        adapter_only_state = bool(renamed_state_dict) and all(
            self._is_adapter_state_key(key) for key in renamed_state_dict.keys()
        )
        if strict and adapter_only_state:
            warnings.warn(
                "Detected adapter-only state dict; loading with strict=False to allow missing base-model weights.",
                RuntimeWarning,
            )
            strict = False

        if not strict:
            return super().load_state_dict(renamed_state_dict, strict=False, assign=assign)

        try:
            return super().load_state_dict(renamed_state_dict, strict=True, assign=assign)
        except RuntimeError:
            pass

        result = super().load_state_dict(renamed_state_dict, strict=False, assign=assign)
        allowed_missing_prefixes = (
            "model.u_scale_embed.",
            "model.time_embed.",
        )
        disallowed_missing = [
            key for key in result.missing_keys
            if not key.startswith(allowed_missing_prefixes)
            and ".adaLN_modulation." not in key
        ]
        if disallowed_missing or result.unexpected_keys:
            problems = []
            if disallowed_missing:
                problems.append(f"Missing key(s): {disallowed_missing}")
            if result.unexpected_keys:
                problems.append(f"Unexpected key(s): {result.unexpected_keys}")
            raise RuntimeError("Error(s) in loading state_dict for "
                               f"{self.__class__.__name__}: " + "; ".join(problems))
        return result

    @staticmethod
    def _is_adapter_state_key(key: str) -> bool:
        # Keep LoRA parameters and project-specific auxiliary embedding weights.
        return (
            ('lora_' in key)
            or ('.u_embed.' in key)
            or ('.u_scale_embed.' in key)
            or ('.time_embed.' in key)
            or ('.adaLN_modulation.' in key)
        )

    def _adapter_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in state_dict.items() if self._is_adapter_state_key(k)}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint.get('state_dict')
        if state_dict is None or self.checkpoint_save_mode == 'full':
            return
        if self.checkpoint_save_mode == 'adapter_only':
            adapter_state_dict = self._adapter_state_dict(state_dict)
            checkpoint['state_dict'] = adapter_state_dict

    def configure_optimizers(self):
        config = self.optim_cfg
        optimizer = {}
        active_parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer["optimizer"] = torch.optim.AdamW(
            active_parameters,
            lr=config["lr"],
        )
        if config.get("lr_scheduler", None) is not None:
            optimizer["lr_scheduler"] = instantiate(
                config["lr_scheduler"],
                optimizer=optimizer["optimizer"]
            )
        if config.get("lr_warmup", 0) > 0:
            warmup_steps = config["lr_warmup"]
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps  # Linear warmup
                else:
                    return 1.0  # keep lr constant after warmup

            optimizer["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer["optimizer"], lr_lambda),
                "interval": "step",
                "frequency": 1,
            }
        return optimizer

    def _log_metrics(self, prefix, metrics, sync_dist=False):
        pbar_metrics = {}
        plot_metrics = {}
        other_metrics = {}
        for key, v in metrics.items():
            prefixed_key = f"{prefix}/{key}"
            if key in self.pbar_metrics:
                pbar_metrics[prefixed_key] = v
            elif (isinstance(v, float) or isinstance(v, int)) or v.shape == torch.Size([]):
                other_metrics[prefixed_key] = v
            else:
                # for idx, vi in enumerate(v):
                #    other_metrics[f"{prefixed_k}_{idx}"] = vi
                # plot_metrics[prefixed_key] = v
                pass
        self.log_dict(pbar_metrics, prog_bar=True, sync_dist=sync_dist)
        for k, v in plot_metrics.items():
            if v.ndim == 1 and self.trainer.logger is not None:
                # Convert to CPU list to prevent GPU memory from being held by wandb
                v_cpu = v.detach().cpu().tolist()
                self.trainer.logger.log_table(key=k, data=list(enumerate(v_cpu)), columns=['position', k])
        self.log_dict(other_metrics, prog_bar=False, sync_dist=sync_dist)

    def training_step(self, batch, batch_idx=None):
        metrics = self.forward(batch, batch_idx)

        loss = metrics['loss']

        metrics_logged = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in metrics.items()}
        metrics_logged['lr'] = self.optimizers().param_groups[0]['lr']
        self._log_metrics('train', metrics_logged)

        return loss

    def on_validation_epoch_start(self):
        self._hist_accumulator = []

    def validation_step(self, batch, batch_idx=None):
        metrics = self.forward(batch, batch_idx, eval=True)
        if 'correct_counts' in metrics:
            self._hist_accumulator.append(metrics['correct_counts'].cpu())
        # Detach all validation metrics (no gradients needed)
        metrics_logged = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in metrics.items()}
        self._log_metrics('val', metrics_logged, sync_dist=True)

    def on_validation_epoch_end(self):
        hist = torch.zeros(21, dtype=torch.float64)
        if self._hist_accumulator:
            counts = torch.cat(self._hist_accumulator).clamp(max=20)
            hist = torch.bincount(counts, minlength=21).double()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            hist = hist.to(self.device)
            torch.distributed.all_reduce(hist)
            hist = hist.cpu()
        if hist.sum() > 0:
            self.hist_base = hist / hist.sum()
        self._hist_accumulator = []

    def adapt_logits(self, logits):
        if self.temperature is not None and self.temperature != 1.0:
            logits = logits / self.temperature
        if self.top_k is not None and self.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, k=self.top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(-1, top_k_indices, top_k_logits)
            logits = mask
        if self.top_p is not None and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        return logits

    def adapt_p(self, p):
        top_k_probs, top_k_indices = torch.topk(p, k=self.top_k, dim=-1)
        # remove additional tokens if top_p is more restrictive
        remove = (top_k_probs.cumsum(dim=-1) - top_k_probs) > self.top_p
        top_k_probs = top_k_probs.masked_fill(remove, 0.0)
        # renormalize; sort by token index so CDF bins match vocab-sorted original
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sort_idx = top_k_indices.argsort(dim=-1)
        return top_k_probs.gather(-1, sort_idx), top_k_indices.gather(-1, sort_idx)

    def forward(self, batch, batch_idx=None, eval=False, return_outputs=False):
        # Unmodified student to get kv-cache
        input_ids = batch['input_ids']
        input_mask = batch['input_mask']

        # Build attention mask and lookup targets for arbitrary position completions
        completion_starts:   Tensor = batch['completion_starts']        # (B, N)
        completion_doc_ids:  Tensor = batch.get('completion_doc_ids')   # (B, N) or None
        doc_ids:      Tensor = batch.get('doc_ids')                     # (B, S) or None
        doc_starts:   Tensor = batch.get('doc_starts')                  # (B, D) or None
        doc_lengths:  Tensor = batch.get('doc_lengths')                 # (B, D) or None
        completion_length: int = batch['completion_length']
        left_bin_edges = batch.get("bin_edges_left")
        right_bin_edges = batch.get("bin_edges_right")

        # AR block mask: causal + document-isolated for packed sequences
        if doc_ids is not None:
            ar_block_mask = create_block_mask(
                make_ar_mask_mod(doc_ids),
                B=input_ids.shape[0], H=None,
                Q_LEN=input_ids.shape[1], KV_LEN=input_ids.shape[1],
                device=input_ids.device,
            )
        else:
            ar_block_mask = None  # standard causal

        ar_outputs = None
        if left_bin_edges is None or right_bin_edges is None:
            left_bin_edges, right_bin_edges, ar_outputs = predict_bin_edges(
                input_ids, input_mask=ar_block_mask,
                model=self.model.ar_forward,
                adapt_logits=self.adapt_logits if (self.temperature is not None or self.top_k is not None or self.top_p is not None) else None,
            )
            left_bin_edges = left_bin_edges.detach()
            right_bin_edges = right_bin_edges.detach()
        nested_batch = self.prepare_nested_batch(
            input_ids, input_mask,
            completion_length, completion_starts,
            left_bin_edges, right_bin_edges,
            eval,
            doc_ids=doc_ids,
            completion_doc_ids=completion_doc_ids,
            doc_starts=doc_starts,
            doc_lengths=doc_lengths,
        )
        attention_mask, auxiliaries, completion_ids, position_ids = nested_batch

        batch_size = input_ids.shape[0]
        num_completions = completion_starts.shape[1]
        ar_outputs, completion_outputs = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            ar_outputs=ar_outputs,
            auxiliaries=auxiliaries.reshape(batch_size, -1),
            auxiliary_position_ids=position_ids.reshape(batch_size, -1),
            auxiliary_mask=attention_mask,
        )

        completion_logits = completion_outputs.logits
        loss_batch_size = batch_size * num_completions * completion_length
        if self.completion_gamma == 1.0:
            completion_loss = torch.nn.functional.cross_entropy(
                completion_logits.reshape(loss_batch_size, -1),
                completion_ids.reshape(loss_batch_size),
                ignore_index=IGNORE_INDEX
            )
        else:
            # Exponential position weights: position k gets weight gamma^k
            gamma_powers = self.completion_gamma ** torch.arange(
                completion_length, device=completion_logits.device, dtype=completion_logits.dtype)
            per_token_loss = torch.nn.functional.cross_entropy(
                completion_logits.reshape(loss_batch_size, -1),
                completion_ids.reshape(loss_batch_size),
                ignore_index=IGNORE_INDEX,
                reduction='none',
            ).reshape(batch_size * num_completions, completion_length)
            valid = completion_ids.reshape(batch_size * num_completions, completion_length) != IGNORE_INDEX
            completion_loss = (per_token_loss * gamma_powers).sum() / \
                (gamma_powers * valid).sum().clamp(min=1e-8)

        # Post-process metrics on flattened completions
        eval_base_shape = (batch_size * num_completions, completion_length)
        completion_logits = completion_logits.reshape(*eval_base_shape, -1)
        metrics = self.compute_sequence_metrics(
            completion_ids.reshape(*eval_base_shape),
            completion_logits.argmax(dim=-1),
            include_outputs=return_outputs,
        )

        loss = 0.0
        if self.completion_loss_weight > 0.0:
            loss = loss + self.completion_loss_weight * completion_loss
        metrics['loss'] = loss
        metrics['l_completion'] = completion_loss
        metrics['num_completions'] = num_completions
        return metrics

    def _make_completion_positions(self, completion_starts: Tensor, completion_length: int,
                                   seq_len: int, device) -> tuple[Tensor, Tensor, Tensor]:
        """Compute position_ids, valid_mask, and safe_positions for all completions."""
        starts = completion_starts.to(device)  # (B, N)
        offsets = torch.arange(completion_length, device=device, dtype=torch.long)  # (L,)
        position_ids = starts[:, :, None] + offsets[None, None, :]  # (B, N, L)
        valid_mask = position_ids < seq_len
        safe_positions = position_ids.clamp(max=seq_len - 1)
        return position_ids, valid_mask, safe_positions

    def _make_completion_block_mask(self, starts: Tensor, completion_length: int, seq_len: int,
                                    batch_size: int, device,
                                    doc_ids: Tensor | None, completion_doc_ids: Tensor | None,
                                    doc_starts: Tensor | None, doc_lengths: Tensor | None):
        """Build the flex_attention block mask for the nested completion batch."""
        num_completions = starts.shape[1]
        total_completion_length = num_completions * completion_length
        # KV layout: [prompt S tokens | completion N*L tokens]
        return create_block_mask(
            make_completion_mask_mod(
                completion_starts=starts,
                completion_doc_ids=completion_doc_ids,
                doc_ids=doc_ids,
                doc_starts=doc_starts,
                doc_lengths=doc_lengths,
                seq_len=seq_len,
                completion_length=completion_length,
            ),
            B=batch_size, H=None,
            Q_LEN=total_completion_length,
            KV_LEN=seq_len + total_completion_length,
            device=device,
        )

    def _gather_completion_ids(self, input_ids: Tensor, safe_positions: Tensor,
                               valid_mask: Tensor, num_completions: int,
                               doc_ids: Tensor | None, completion_doc_ids: Tensor | None,
                               starts: Tensor) -> Tensor:
        """Gather target token IDs, masking out-of-bounds and cross-document positions."""
        completion_ids = torch.gather(
            input_ids[:, None, :].expand(-1, num_completions, -1),
            2, safe_positions,
        )
        completion_ids[~valid_mask] = IGNORE_INDEX

        if doc_ids is not None:
            target_doc_ids = torch.gather(
                doc_ids[:, None, :].expand(-1, num_completions, -1),
                2, safe_positions,
            )  # (B, N, L)
            start_doc_ids = completion_doc_ids if completion_doc_ids is not None \
                else torch.gather(doc_ids, 1, starts)  # (B, N)
            cross_doc_mask = target_doc_ids != start_doc_ids[:, :, None]  # (B, N, L)
            first_cross_mask = cross_doc_mask & (cross_doc_mask.cumsum(dim=-1) == 1)
            eos_token_id = getattr(self.model.tokenizer, 'eos_token_id', None)
            completion_ids[first_cross_mask] = eos_token_id if eos_token_id is not None else IGNORE_INDEX
            completion_ids[cross_doc_mask & ~first_cross_mask] = IGNORE_INDEX

        return completion_ids

    def _gather_bin_edges(self, left_bin_edges: Tensor, right_bin_edges: Tensor,
                          safe_positions: Tensor, valid_mask: Tensor,
                          num_completions: int) -> tuple[Tensor, Tensor]:
        """Gather bin edges at completion positions (shifted by -1 for logit alignment)."""
        edge_positions = (safe_positions - 1).clamp(min=0, max=left_bin_edges.shape[1] - 1)
        all_left = torch.gather(left_bin_edges[:, None, :].expand(-1, num_completions, -1), 2, edge_positions)
        all_right = torch.gather(right_bin_edges[:, None, :].expand(-1, num_completions, -1), 2, edge_positions)
        all_left[~valid_mask] = 0.0
        all_right[~valid_mask] = 0.0
        return all_left, all_right

    def prepare_nested_batch(self, input_ids: Tensor, input_mask: Tensor,
                             completion_length: int, completion_starts: Tensor,
                             left_bin_edges: Tensor | Any, right_bin_edges: Tensor | Any,
                             eval: bool,
                             doc_ids: Tensor | None = None,
                             completion_doc_ids: Tensor | None = None,
                             doc_starts: Tensor | None = None,
                             doc_lengths: Tensor | None = None,
                             ):
        if left_bin_edges is None or right_bin_edges is None:
            raise ValueError("left_bin_edges and right_bin_edges must be provided")
        assert left_bin_edges.shape[0] == input_ids.shape[0], \
            f"bin_edges batch dim {left_bin_edges.shape[0]} != input_ids {input_ids.shape[0]}"
        # bin_edges may be (B, S) or (B, S-1) depending on source
        assert left_bin_edges.shape[1] >= input_ids.shape[1] - 1, \
            f"bin_edges length {left_bin_edges.shape[1]} < seq_len-1 {input_ids.shape[1] - 1}"

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        starts = completion_starts.to(device)  # (B, N)
        num_completions = starts.shape[1]
        assert (starts > 0).all(), "Completion start indices must be > 0 to have bin edges"

        position_ids, valid_mask, safe_positions = self._make_completion_positions(
            starts, completion_length, seq_len, device)
        block_mask = self._make_completion_block_mask(
            starts, completion_length, seq_len, batch_size, device,
            doc_ids, completion_doc_ids, doc_starts, doc_lengths)
        completion_ids = self._gather_completion_ids(
            input_ids, safe_positions, valid_mask, num_completions,
            doc_ids, completion_doc_ids, starts)
        all_left_edges, all_right_edges = self._gather_bin_edges(
            left_bin_edges, right_bin_edges, safe_positions, valid_mask, num_completions)

        position_ids[~valid_mask] = 0
        auxiliaries = self.sample_auxiliaries(all_left_edges, all_right_edges, eval)
        return block_mask, auxiliaries, completion_ids, position_ids

    def sample_auxiliaries(self, left_bin_edges: torch.Tensor, right_bin_edges: torch.Tensor,
                           eval: bool) -> torch.Tensor:
        device = left_bin_edges.device
        if not eval:
            beta_concentration = torch.tensor(0.3, device=device, dtype=torch.float32)
            if device.type == 'mps':
                beta_concentration = beta_concentration.cpu()
            z_rnd = torch.distributions.Beta(beta_concentration, beta_concentration).sample(
                left_bin_edges.shape).to(device)
        else:
            z_rnd = torch.rand(left_bin_edges.shape, device=device, dtype=torch.float32)
        auxiliaries = left_bin_edges + (right_bin_edges - left_bin_edges) * z_rnd
        return auxiliaries

    @torch.no_grad()
    def compute_sequence_metrics(self, completion_ids, student_predicted, include_outputs=False):
        mask = completion_ids != IGNORE_INDEX
        identical = (completion_ids == student_predicted) & mask
        count_per_length = (mask.long().sum(dim=0) + 1e-8)
        acc_per_position = identical.long().sum(dim=0) / count_per_length
        accuracy = identical.sum() / (mask.sum() + 1e-8)

        metrics = {
            'accuracy': accuracy,
            'acc_per_position': acc_per_position,
            'completion_length': mask.sum(dim=1).float().mean(),
        }

        # Correct count before first error
        correct_counts = torch.where(
            # all tokens correct?
            ((completion_ids == student_predicted) | ~mask).all(dim=1),
            mask.long().sum(1),
            ((completion_ids == student_predicted) & mask).float().argmin(dim=1),
        ).long()
        metrics["correct"] = correct_counts.float().mean()
        metrics["correct_counts"] = correct_counts

        for positional_metric in ['acc_per_position']:
            if positional_metric in metrics:
                short_positional_metric = positional_metric.replace('_per_position', '')
                for position in range(min(10, metrics[positional_metric].shape[0])):
                    metrics[f'{short_positional_metric}_pos_{position}'] = metrics[positional_metric][position]
        if include_outputs:
            metrics['outputs'] = student_predicted
        return metrics

    def proposals(self, num_tokens=None, student_p=None, n_verify=None, double_at=100, metrics=None):
        """
        Optimize proposals B wrt overhead adjusted expected # correct tokens
        max_B [sum_i A_i * H(B_i)] / [1 + sum_i B_i / 50)]

        If A_0 = 1 this becomes max_k H(k) / [1 + k / 50] = 14
        """
        assert self.hist_base is not None, "hist_base must be provided to use proposals()"
        # Estimated probability of # correct tokens
        if student_p is None:
            A = self.hist_base
            if n_verify is not None:
                A = torch.cat([A[:n_verify], torch.tensor([A[n_verify:].sum()])])
        else:
            # assert student_p.shape[1] == n_verify
            A = torch.ones(student_p.shape[1] + 1)
            A[1:] = torch.cumprod(student_p[0].cpu(), dim=-1)
            A[:-1] *= 1 - student_p[0].cpu()
        # Reward; Estimated # correct tokens in the next step given k proposed tokens
        arange_21 = torch.arange(21)
        H_hist = torch.cumsum(self.hist_base * arange_21, dim=-1)
        # or: # proposed tokens
        H_count = arange_21
        # switch dynamically
        # if metrics is not None and len(metrics['Nrel']) > 0:
        #     rho = np.clip((np.array(metrics['Nrel']).mean() - 0.2) / 0.2, 0, 1)
        # else:
        #     rho = 0
        rho = 0
        H = (1 - rho) * H_hist + rho * H_count
        # A = A.clip(min=0.05)

        M = self.tokens_per_student_call
        arange_Mp1 = torch.arange(M + 1)
        AH = A[:, None] * H[None, :M + 1]  # [n_pos, M+1]
        inf = torch.tensor(float('inf'))

        if num_tokens is None:
            # Optimize based on per-token cost of 1/<double_at>
            lam = 0
            for _ in range(3):
                B = torch.argmax(AH - lam * arange_Mp1[None, :] / double_at, dim=1)
                lam_pre = lam
                lam = torch.sum(A * H[B]) / (1 + B.sum() / double_at)
                if (lam_pre == lam).all():
                    break
            return B.tolist()
        else:
            num_tokens = min(num_tokens, M * A.shape[0])

            # --- Binary search on lambda --- faster at first
            # lam_lo = 0.0
            # lam_hi = float(A.max())
            # for _ in range(50):
            #     lam = (lam_lo + lam_hi) / 2
            #     B = torch.argmax(AH - lam * self.arange_tpscp1[None, :], dim=1)
            #     total = B.sum().item()
            #     if total > num_tokens:
            #         lam_lo = lam
            #     elif total < num_tokens:
            #         lam_hi = lam
            #     else:
            #         break
            #     if abs(total - num_tokens) <= 5:
            #         break

            # Greedy estimate of lam
            A_idx = torch.argsort(A, descending=True)
            R = num_tokens // M
            r = num_tokens % M
            B = torch.zeros([A.shape[0]], dtype=int)
            B[A_idx[:R]] = M
            B[A_idx[R]] = r
            if r > 0:
                lam = float(A[A_idx[R]] * (H[r] - H[r - 1]))
            else:
                lam = float(A[A_idx[R - 1]] * (H[M] - H[M - 1]))
            B = torch.argmax(AH - lam * arange_Mp1[None, :], dim=1)

            # --- Greedy correction ---
            # B = B.clone()
            total = B.sum().item()

            while total > num_tokens:
                # Decrement the position with the smallest marginal gain of its last token
                # marginal gain of token b_i: A_i * (H[b_i] - H[b_i-1])
                can_dec = B > 0
                gains = torch.where(can_dec, A * (H[B] - H[(B - 1).clamp(min=1)]), inf)
                idx = torch.argmin(gains)
                B[idx] -= 1
                total -= 1

            while total < num_tokens:
                # Increment the position with the highest marginal gain of the next token
                # marginal gain of token b_i+1: A_i * (H[b_i+1] - H[b_i])
                can_inc = B < M
                gains = torch.where(can_inc, (A * (H[(B + 1).clamp(max=M)] - H[B])).clamp(min=1e-10), -inf)
                idx = torch.argmax(gains)
                B[idx] += 1
                total += 1

            return B.tolist()

    @torch.inference_mode()
    def generate(self, batch, max_new_tokens, return_metrics=False, return_past_key_values=False,
                 eos=None, fixed_tokens=True, pad_token=13, past_kv_cache=None, callback=None,
                 **kwargs):
        """
        Partial Quadratic Coding using kv-cached Gated LoRA

        Input:
        ------
        fixed_tokens           - if True, force each transformer call to use the same number of
                                 verifying and proposed tokens, padding if necessary.
        past_kv_cache          - optional (prompt_ids, DynamicCache) from a previous generate()
                                 call for prompt-prefix reuse.
        return_past_key_values - if True, include (prompt_ids, DynamicCache) in the return value
                                 so it can be passed as past_kv_cache on the next call.
        """
        prompt_ids = batch['prompt_ids']
        assert prompt_ids.shape[0] == 1, "Batch size must be 1"
        assert self.model.inference_mode, "Call enter_inference_mode() before generate()"
        assert self.top_k is not None and self.top_k > 0
        assert self.top_p is not None and self.top_p < 1.0
        dev = prompt_ids.device
        tpsc = self.tokens_per_student_call
        ones_tpsc = torch.ones([1, tpsc], dtype=torch.long, device=dev)
        metrics = {'correct': [], 'N': [], 'Nrel': []}

        # Number of tokens proposed per speculative decoding step
        num_proposed_tokens = (self.total_token_budget or tpsc) if fixed_tokens else None

        # Verify in parallel
        tokens_to_fill = max_new_tokens
        tokens_to_verify = max_new_tokens
        z_rnd_all = torch.rand([prompt_ids.shape[0], max_new_tokens + tpsc + (num_proposed_tokens if fixed_tokens else 0)], device=dev, dtype=torch.float32)
        if fixed_tokens:
            n_props = ((num_proposed_tokens // tpsc) * [tpsc] + [(num_proposed_tokens % tpsc)] + tpsc * [0])[:tpsc]
        else:
            n_props = self.proposals(n_verify=0, num_tokens=num_proposed_tokens)
        
        if eos is None:
            eos = getattr(self.model.tokenizer, 'eos_token_id', None)
            if eos is None:
                raise ValueError("eos token id must be provided either via model.tokenizer.eos_token_id or the eos argument")

        # Reuse cached KV state for a matching prompt prefix
        if past_kv_cache is not None:
            past_prompt_ids, cache = past_kv_cache
            end = min(prompt_ids.shape[1], past_prompt_ids.shape[1])
            match = prompt_ids[:, :end] == past_prompt_ids[:, :end]
            keep = end if match.all() else match.float().argmin().item()
            cache.crop(keep)
            kv_cache = cache
        else:
            kv_cache = DynamicCache()
        if not fixed_tokens:
            self.model.set_gate_window(0)
        # Prefill: process prompt tokens (without proposing new ones)
        outputs = self.model.inference_forward(
            input_ids=prompt_ids[:, kv_cache.get_seq_length():-1],
            auxiliaries=z_rnd_all[:, :num_proposed_tokens if fixed_tokens else 0],
            past_key_values=kv_cache,
            use_cache=True,
        )
        kv_cache = outputs.past_key_values
        if fixed_tokens:
            # Pad to length
            kv_cache.crop(prompt_ids.shape[1] - 1)
            prompt_ids = torch.cat([
                prompt_ids,
                pad_token * ones_tpsc[:, :self.tokens_per_student_call - 1]
            ], dim=1)
            tokens_to_fill -= self.tokens_per_student_call - 1
        # torch.cuda.synchronize()
        # timing['call'] += [time.time() - scall]

        while tokens_to_verify > 0:
            if not fixed_tokens:
                n_props = [min(n, max(0, tokens_to_verify - d)) for d, n in enumerate(n_props)]
            n_verify = tokens_to_verify - tokens_to_fill
            # assert n_verify == len(n_props) - 1
            seq_len = prompt_ids.shape[1] + sum(n_props)
            pos = prompt_ids.shape[1] - kv_cache.get_seq_length()
            metrics['N'] += [pos + sum(n_props)]

            # Prepare inputs
            z_idx = max_new_tokens - tokens_to_verify
            z_rnd = torch.cat([z_rnd_all[:, z_idx + d:z_idx + d + n_prop] for d, n_prop in enumerate(n_props)], dim=1)
            input_ids = prompt_ids[:, kv_cache.get_seq_length():]
            K = kv_cache.get_seq_length()
            P = prompt_ids.shape[1]
            Q_LEN = seq_len - K
            # Position IDs: proposals for hypothesis d are shifted to position P-n_verify+d-1
            input_position_ids = torch.arange(K, seq_len, device=dev)[None, :]
            midx = pos
            for d, n_prop in enumerate(n_props):
                input_position_ids[:, midx: midx + n_prop] -= midx - pos + n_verify - d + 1
                midx += n_prop
            # Dense additive attention mask [1, 1, Q_LEN, KV_LEN].
            # SDPA handles variable KV_LEN natively without per-shape recompilation.
            midx = pos
            input_mask = torch.tril(torch.ones(Q_LEN, seq_len, device=dev), diagonal=K)
            for d, n_prop in enumerate(n_props):
                input_mask[midx: midx + n_prop, P - n_verify + d : K + midx] = 0
                midx += n_prop
            input_mask = (1 - input_mask[None, None].to(torch.float16)) * -1e15

            # Student proposals
            if not fixed_tokens:
                self.model.set_gate_window(sum(n_props))
            outputs = self.model.inference_forward(
                input_ids=input_ids,
                attention_mask=input_mask,
                position_ids=input_position_ids,
                auxiliaries=z_rnd,
                past_key_values=kv_cache,
                use_cache=True
            )

            kv_cache = outputs.past_key_values
            full_logits = outputs.logits
            student_logits = full_logits[:, pos:]
            if self.temperature is not None and self.temperature != 1.0:
                full_logits[:, :pos] = full_logits[:, :pos] / self.temperature
            full_p = torch.softmax(full_logits, dim=-1)
            student_p = full_p[:, pos:]
            tgt_p, tgt_indices = self.adapt_p(full_p[:, :pos])
            student_predicted = student_logits.argmax(dim=-1)
            student_p_max = student_p.gather(-1, student_predicted[..., None])[..., 0]

            # Verify last speculated tokens
            if n_verify > 0:
                # assert n_verify == tgt_logits.shape[1] - 1
                # with record_function("loop: verify + accept"):
                right_bin_edges = tgt_p.cumsum(dim=-1)  # [1, n_verify+1, top_k]
                right_bin_edges[..., -1] = 1
                z_idx = max_new_tokens - tokens_to_verify
                z_rnd = z_rnd_all[:, z_idx:z_idx + n_verify + 1]
                bin_idx = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
                correct_tokens = tgt_indices.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
                check_tokens = correct_tokens[:, :-1]
                predict_tokens = prompt_ids[..., - n_verify:]
                matches = (predict_tokens == check_tokens)
                num_correct = matches.float().argmin(dim=1)
                num_correct[matches.all(dim=1)] = n_verify
                num_correct = int(num_correct[0])
                num_new = num_correct + 1
                tokens_to_verify -= num_correct
                kv_cache.crop(prompt_ids.shape[-1] - (n_verify - num_correct))
                prev_prop = sum(n_props[:num_correct])
                ths_student_predicted = student_predicted[:, prev_prop: prev_prop + n_props[num_correct]]
                ths_student_p = student_p_max[:, prev_prop: prev_prop + n_props[num_correct]]
                metrics['Nrel'] += [num_correct / n_verify]
                # Verify first speculated and add other speculated tokens
                if fixed_tokens and ths_student_predicted.shape[1] < self.tokens_per_student_call:
                    # Pad with linbebreaks
                    ths_student_predicted = torch.cat([ths_student_predicted, pad_token * ones_tpsc[:, :self.tokens_per_student_call - ths_student_predicted.shape[1]]], dim=1)
                    ths_student_p = torch.cat([ths_student_p, 0 * ones_tpsc[:, :self.tokens_per_student_call - ths_student_p.shape[1]]], dim=1)
                if ths_student_predicted.shape[1] == 0:
                    match = False
                else:
                    ths_student_predicted[0, 0] = correct_tokens[0, num_correct]
                    match = ths_student_predicted[0, 0] == correct_tokens[0, num_correct]
                if not match:
                    # Discard speculated tokens
                    prompt_ids = torch.cat([
                        prompt_ids[:, :prompt_ids.shape[1] - n_verify],
                        correct_tokens[:, :num_correct + 1],
                    ], dim=1)
                    tokens_to_verify -= 1
                    tokens_to_fill = tokens_to_verify
                    n_props = self.proposals(n_verify=0, num_tokens=num_proposed_tokens, metrics=metrics)
                    if callback is not None:
                        callback(prompt_ids[0], prompt_ids.shape[1])
                else:
                    # Add new speculated tokens
                    prompt_ids = torch.cat([
                        prompt_ids[:, :prompt_ids.shape[1] - (n_verify - num_correct)],
                        ths_student_predicted,
                    ], dim=1)
                    tokens_to_fill = tokens_to_verify - ths_student_predicted.shape[1]
                    tokens_to_verify -= 1
                    n_props = self.proposals(student_p=ths_student_p[:, 1:], num_tokens=num_proposed_tokens, metrics=metrics)
                    if callback is not None:
                        callback(prompt_ids[0], prompt_ids.shape[1] - ths_student_predicted.shape[1] + 1)
                metrics['correct'] += [num_new]
                if eos in correct_tokens[:, :num_correct + 1]:
                    break
            else:
                # Only relevant if fixed_tokens=False and we don't force correct tokens
                raise NotImplementedError
            # else:
            #     # Accept one token
            #     tgt_logits = self.adapt_logits(tgt_logits[:, -1:])
            #     right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
            #     right_bin_edges[..., -1] = 1
            #     right_bin_edges[..., 0] = 0
            #     z_idx = max_new_tokens - tokens_to_verify
            #     z_rnd = z_rnd_all[:, z_idx:z_idx + 1]
            #     correct_token = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
            #     kv_cache.crop(int(prompt_ids.shape[-1]))
            #
            #     if fixed_tokens and student_predicted.shape[1] < self.tokens_per_student_call:
            #         # Pad with linbebreaks
            #         student_predicted = torch.cat([student_predicted, 13 * torch.ones([1, self.tokens_per_student_call - ths_student_predicted.shape[1]], dtype=int, device=device)], dim=1)
            #         student_p_max = torch.cat([student_p_max, 13 * torch.ones([1, self.tokens_per_student_call - ths_student_p.shape[1]], device=device)], dim=1)
            #     if student_predicted.shape[1] == 0:
            #         match = False
            #     else:
            #         student_predicted[0, 0] = correct_token[0, 0]
            #         match = student_predicted[0, 0] == correct_token[0, 0]
            #     if not match:
            #         # Discard speculated tokens
            #         prompt_ids = torch.cat([
            #             prompt_ids,
            #             correct_token,
            #         ], dim=1)
            #         tokens_to_verify -= 1
            #         tokens_to_fill = tokens_to_verify
            #         n_props = self.proposals(n_verify=0, num_tokens=num_proposed_tokens, metrics=metrics)
            #     else:
            #         # Add new speculated tokens
            #         prompt_ids = torch.cat([
            #             prompt_ids,
            #             student_predicted[:, :n_props[0]],
            #         ], dim=1)
            #         tokens_to_fill = tokens_to_verify - n_props[0]
            #         tokens_to_verify -= 1
            #         n_props = self.proposals(student_p=student_p_max[:, 1:], num_tokens=num_proposed_tokens, metrics=metrics)
            #     metrics['correct'] += [1]
            #     if eos == correct_token[0, 0]:
            #         break

            # torch.cuda.synchronize()
            # timing['step'] += [time.time() - s]

        # print(1000 * np.mean(timing['call'][1:]), 1000 * timing['call'][0], 1000 * np.mean(timing['step']))
        # plt.scatter(metrics['off'], metrics['offp'], s=2, alpha=0.5); plt.gca().set(xlabel='If the predicted token is wrong, the k-th one after is', ylabel='student confidence for actual correct token'); plt.show()
        metrics = {
            'completion': prompt_ids,
            'correct_per_call': np.mean(metrics['correct']),
            'correct_all': metrics['correct'],
            'num_calls': len(metrics['correct']),
        }

        if return_past_key_values and return_metrics:
            return prompt_ids, (prompt_ids, kv_cache), metrics
        if return_past_key_values:
            return prompt_ids, (prompt_ids, kv_cache)
        if return_metrics:
            return prompt_ids, metrics
        return prompt_ids