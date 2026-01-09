"""
finetune_soft_target.py

Parameter-efficient fine-tuning of OpenVLA models with alternative loss functions.
Instead of hard cross-entropy loss (only correct bin gets weight 1), we provide:

    1. soft_ce (default): Gaussian soft target distribution centered on the correct bin,
       allowing the model to learn that nearby bins represent similar actions.

    2. wasserstein: 1D Wasserstein distance (Earth Mover's Distance) using closed-form
       CDF solution. Considers the geometry of the action space.

    3. sinkhorn: Entropy-regularized optimal transport distance. More general but
       computationally heavier.

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune_soft_target.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune_soft_target.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    --loss_type soft_ce \
                                    --soft_target_sigma 2.0 \
                                    ...

    - [Wasserstein Loss]: torchrun ... vla-scripts/finetune_soft_target.py --loss_type wasserstein ...
    - [Sinkhorn Loss]: torchrun ... vla-scripts/finetune_soft_target.py --loss_type sinkhorn --sinkhorn_epsilon 0.1 ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.nn.functional as F
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_gaussian_soft_target(
    target_indices: torch.Tensor,
    num_classes: int,
    sigma: float = 2.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create Gaussian soft target distribution centered on the target index.

    Args:
        target_indices: Shape [N] - indices of target classes (action token indices within action vocab)
        num_classes: Total number of action bins (e.g., 256)
        sigma: Standard deviation of Gaussian (in bin units)
        device: Device to create tensor on

    Returns:
        Soft target distribution of shape [N, num_classes]
    """
    if device is None:
        device = target_indices.device

    N = target_indices.shape[0]

    # Create bin indices [0, 1, ..., num_classes-1]
    bin_indices = torch.arange(num_classes, device=device, dtype=torch.float32)  # [num_classes]

    # Expand for broadcasting: target_indices [N, 1], bin_indices [1, num_classes]
    target_indices = target_indices.float().unsqueeze(1)  # [N, 1]
    bin_indices = bin_indices.unsqueeze(0)  # [1, num_classes]

    # Compute Gaussian weights: exp(-0.5 * ((bin - target) / sigma)^2)
    distances = (bin_indices - target_indices) / sigma  # [N, num_classes]
    soft_targets = torch.exp(-0.5 * distances ** 2)  # [N, num_classes]

    # Normalize to create probability distribution
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

    return soft_targets


def soft_cross_entropy_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute soft cross-entropy loss (KL divergence with soft targets).

    Args:
        logits: Raw logits of shape [N, num_classes]
        soft_targets: Soft target distribution of shape [N, num_classes]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Soft cross-entropy loss
    """
    # Compute log softmax of logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Cross-entropy with soft targets: -sum(p * log(q))
    loss = -torch.sum(soft_targets * log_probs, dim=-1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def wasserstein_1d_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    n_bins: int = 256,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance) between predicted distribution and target.

    For 1D distributions, Wasserstein-1 has a closed-form solution using CDFs:
        W_1(P, Q) = sum(|CDF_P - CDF_Q|)

    This loss considers the geometry of the action space - nearby bins are treated as more similar.

    Args:
        logits: Raw logits of shape [N, n_bins]
        target_indices: Target bin indices of shape [N]
        n_bins: Number of action bins
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Wasserstein-1 distance loss
    """
    # Convert logits to probabilities
    pred_probs = F.softmax(logits, dim=-1)  # [N, n_bins]

    # Create one-hot target distribution
    target_probs = torch.zeros_like(pred_probs)
    target_probs.scatter_(1, target_indices.unsqueeze(1), 1.0)

    # Compute CDFs (cumulative distribution functions)
    pred_cdf = torch.cumsum(pred_probs, dim=-1)
    target_cdf = torch.cumsum(target_probs, dim=-1)

    # Wasserstein-1 distance = sum of absolute CDF differences
    # Normalize by n_bins to keep loss scale similar to cross-entropy
    wasserstein = torch.sum(torch.abs(pred_cdf - target_cdf), dim=-1) / n_bins

    if reduction == "mean":
        return wasserstein.mean()
    elif reduction == "sum":
        return wasserstein.sum()
    else:
        return wasserstein


def sinkhorn_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    n_bins: int = 256,
    epsilon: float = 0.1,
    n_iters: int = 50,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Sinkhorn distance (entropy-regularized Wasserstein) between predicted distribution and target.

    Sinkhorn provides a differentiable approximation to optimal transport distance.
    For 1D case, this is more expensive than closed-form Wasserstein but more general.

    Args:
        logits: Raw logits of shape [N, n_bins]
        target_indices: Target bin indices of shape [N]
        n_bins: Number of action bins
        epsilon: Entropy regularization parameter (smaller = closer to true Wasserstein)
        n_iters: Number of Sinkhorn iterations
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Sinkhorn distance loss
    """
    N = logits.shape[0]
    device = logits.device

    # Convert logits to probabilities
    pred_probs = F.softmax(logits, dim=-1)  # [N, n_bins]

    # Create one-hot target distribution
    target_probs = torch.zeros_like(pred_probs)
    target_probs.scatter_(1, target_indices.unsqueeze(1), 1.0)

    # Add small epsilon to avoid numerical issues
    pred_probs = pred_probs + 1e-8
    pred_probs = pred_probs / pred_probs.sum(dim=-1, keepdim=True)
    target_probs = target_probs + 1e-8
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)

    # Cost matrix: |i - j| / n_bins for normalized 1D distance
    indices = torch.arange(n_bins, device=device, dtype=torch.float32)
    cost_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)) / n_bins  # [n_bins, n_bins]

    # Gibbs kernel
    K = torch.exp(-cost_matrix / epsilon)  # [n_bins, n_bins]

    # Sinkhorn iterations
    # u, v are scaling vectors
    u = torch.ones(N, n_bins, device=device)
    v = torch.ones(N, n_bins, device=device)

    for _ in range(n_iters):
        # u = pred_probs / (K @ v)
        u = pred_probs / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        # v = target_probs / (K.T @ u)
        v = target_probs / (torch.matmul(K.T, u.unsqueeze(-1)).squeeze(-1) + 1e-8)

    # Compute transport plan: T = diag(u) @ K @ diag(v)
    # Sinkhorn distance = <T, C> = sum(T * C)
    # T[i,j] = u[i] * K[i,j] * v[j]
    transport = u.unsqueeze(-1) * K.unsqueeze(0) * v.unsqueeze(-2)  # [N, n_bins, n_bins]
    sinkhorn_dist = (transport * cost_matrix.unsqueeze(0)).sum(dim=(-1, -2))  # [N]

    if reduction == "mean":
        return sinkhorn_dist.mean()
    elif reduction == "sum":
        return sinkhorn_dist.sum()
    else:
        return sinkhorn_dist


def compute_soft_target_action_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_token_begin_idx: int,
    vocab_size: int,
    num_image_patches: int,
    n_bins: int = 256,
    loss_type: str = "soft_ce",
    sigma: float = 2.0,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iters: int = 50,
) -> tuple[torch.Tensor, int]:
    """
    Compute action loss for action tokens only with various loss types.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size] (includes image patches)
        labels: Target labels [batch, text_seq_len] (-100 for ignore, no image patches)
        action_token_begin_idx: Start index of action tokens in vocabulary
        vocab_size: Total vocabulary size
        num_image_patches: Number of image patch tokens to skip in logits
        n_bins: Number of action bins
        loss_type: Type of loss function to use
            - "soft_ce": Soft cross-entropy with Gaussian targets (default)
            - "wasserstein": 1D Wasserstein distance (Earth Mover's Distance)
            - "sinkhorn": Sinkhorn distance (entropy-regularized OT)
        sigma: Gaussian sigma for soft targets (only used when loss_type="soft_ce")
        sinkhorn_epsilon: Entropy regularization for Sinkhorn (only used when loss_type="sinkhorn")
        sinkhorn_iters: Number of Sinkhorn iterations (only used when loss_type="sinkhorn")

    Returns:
        Tuple of (loss, num_action_tokens)
    """
    # Slice logits to skip image patches and align with labels
    # logits[:, num_patches:-1] predicts labels[:, 1:]
    # (The last logit predicts next token after sequence, which we don't have label for)
    text_logits = logits[:, num_image_patches:-1, :].contiguous()  # [batch, text_seq_len-1, logits_vocab_size]
    shift_labels = labels[:, 1:].contiguous()  # [batch, text_seq_len-1]

    # Get actual logits vocab size (may be padded, e.g., 32064 vs tokenizer's 32000)
    logits_vocab_size = text_logits.shape[-1]

    # Flatten
    text_logits = text_logits.view(-1, logits_vocab_size)  # [batch * (text_seq_len-1), logits_vocab_size]
    shift_labels = shift_labels.view(-1)  # [batch * (text_seq_len-1)]

    # Find action token positions (labels that are action tokens, not -100)
    action_mask = shift_labels > action_token_begin_idx

    if action_mask.sum() == 0:
        # No action tokens in this batch - return zero loss connected to computation graph
        return (logits.sum() * 0.0), 0

    # Extract action token logits and labels
    action_logits = text_logits[action_mask]  # [num_action_tokens, vocab_size]
    action_labels = shift_labels[action_mask]  # [num_action_tokens]

    # Convert action token IDs to action bin indices (0 to n_bins-1)
    # ActionTokenizer encodes: token_id = vocab_size - discretized (discretized is 1 to n_bins)
    # So to get bin index: bin_idx = vocab_size - token_id - 1
    # This means: token (vocab_size - 1) → bin 0, token (vocab_size - n_bins) → bin (n_bins - 1)
    action_bin_indices = vocab_size - action_labels - 1

    # Clamp to valid range (handles edge cases)
    action_bin_indices = torch.clamp(action_bin_indices, 0, n_bins - 1)

    # Extract only action token logits from positions (vocab_size - n_bins) to (vocab_size - 1)
    # Note: We use tokenizer's vocab_size, not logits_vocab_size (which may be padded)
    # Action tokens are at vocab_size - n_bins (bin 255) to vocab_size - 1 (bin 0)
    # We flip to get logits ordered by bin index: logit[0] = bin 0, logit[255] = bin 255
    action_token_logits = action_logits[:, vocab_size - n_bins : vocab_size].flip(dims=[-1])  # [num_action_tokens, n_bins]

    # Compute loss based on loss_type
    if loss_type == "soft_ce":
        # Soft cross-entropy with Gaussian targets
        soft_targets = create_gaussian_soft_target(
            action_bin_indices,
            num_classes=n_bins,
            sigma=sigma,
            device=action_logits.device,
        )
        loss = soft_cross_entropy_loss(action_token_logits, soft_targets, reduction="mean")

    elif loss_type == "wasserstein":
        # 1D Wasserstein distance (closed-form solution using CDFs)
        loss = wasserstein_1d_loss(
            action_token_logits,
            action_bin_indices,
            n_bins=n_bins,
            reduction="mean",
        )

    elif loss_type == "sinkhorn":
        # Sinkhorn distance (entropy-regularized optimal transport)
        loss = sinkhorn_loss(
            action_token_logits,
            action_bin_indices,
            n_bins=n_bins,
            epsilon=sinkhorn_epsilon,
            n_iters=sinkhorn_iters,
            reduction="mean",
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'soft_ce', 'wasserstein', 'sinkhorn'.")

    return loss, action_mask.sum().item()


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # Loss Function Parameters
    loss_type: str = "soft_ce"                                      # Loss function type: "soft_ce", "wasserstein", "sinkhorn"
                                                                    #   soft_ce: Gaussian soft cross-entropy (default)
                                                                    #   wasserstein: 1D Wasserstein distance (EMD)
                                                                    #   sinkhorn: Sinkhorn distance (regularized OT)

    # Soft CE Parameters (only used when loss_type="soft_ce")
    soft_target_sigma: float = 2.0                                  # Gaussian sigma for soft targets (in bin units)
                                                                    #   Higher = softer targets, more tolerance for nearby bins
                                                                    #   sigma=1.0 means ~68% weight within +/- 1 bin
                                                                    #   sigma=2.0 means ~68% weight within +/- 2 bins

    # Sinkhorn Parameters (only used when loss_type="sinkhorn")
    sinkhorn_epsilon: float = 0.1                                   # Entropy regularization (smaller = closer to true Wasserstein)
    sinkhorn_iters: int = 50                                        # Number of Sinkhorn iterations

    # Accuracy Margin Parameter
    action_accuracy_margin: int = 0                                 # Margin for action accuracy calculation (in bins)
                                                                    #   0 = strict exact match (default, same as original)
                                                                    #   3 = predictions within +/- 3 bins are correct

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # Print loss type specific info
    if cfg.loss_type == "soft_ce":
        loss_info = f"Soft CE Loss (sigma={cfg.soft_target_sigma})"
    elif cfg.loss_type == "wasserstein":
        loss_info = "Wasserstein-1D Loss"
    elif cfg.loss_type == "sinkhorn":
        loss_info = f"Sinkhorn Loss (epsilon={cfg.sinkhorn_epsilon}, iters={cfg.sinkhorn_iters})"
    else:
        loss_info = f"Unknown Loss ({cfg.loss_type})"
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}` with {loss_info}")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    # Add loss type specific suffix
    if cfg.loss_type == "soft_ce":
        exp_id += f"+soft_ce-sigma{cfg.soft_target_sigma}"
    elif cfg.loss_type == "wasserstein":
        exp_id += "+wasserstein"
    elif cfg.loss_type == "sinkhorn":
        exp_id += f"+sinkhorn-eps{cfg.sinkhorn_epsilon}"
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Get vocab size for loss computation
    # IMPORTANT: Use tokenizer's vocab_size (not model config) to match ActionTokenizer
    # The model config may have padded vocab_size, but action tokens use tokenizer's vocab_size
    vocab_size = processor.tokenizer.vocab_size

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Get number of image patches for loss computation (after DDP wrapping)
    num_image_patches = vla.module.vision_backbone.featurizer.patch_embed.num_patches

    # Load Fine-tuning Dataset
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Forward pass (don't use output.loss, compute custom soft target loss)
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],  # Still pass labels for output.loss comparison
                )

                # Compute action loss (soft_ce, wasserstein, or sinkhorn)
                loss, num_action_tokens = compute_soft_target_action_loss(
                    logits=output.logits,
                    labels=batch["labels"].to(device_id),
                    action_token_begin_idx=action_tokenizer.action_token_begin_idx,
                    vocab_size=vocab_size,
                    num_image_patches=num_image_patches,
                    n_bins=action_tokenizer.n_bins,
                    loss_type=cfg.loss_type,
                    sigma=cfg.soft_target_sigma,
                    sinkhorn_epsilon=cfg.sinkhorn_epsilon,
                    sinkhorn_iters=cfg.sinkhorn_iters,
                )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, num_image_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy (with optional margin tolerance)
            # margin=0: exact match only (default, same as original)
            # margin=3: predictions within +/- 3 bins are considered correct
            if cfg.action_accuracy_margin == 0:
                correct_preds = (action_preds == action_gt) & mask
            else:
                pred_diff = torch.abs(action_preds.long() - action_gt.long())
                correct_preds = (pred_diff <= cfg.action_accuracy_margin) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                        "hard_ce_loss": output.loss.item(),  # Also log original hard CE loss for comparison
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0 and gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            merged_vla.save_pretrained(run_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
