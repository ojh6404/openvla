"""
finetune_soft_target.py

Parameter-efficient fine-tuning of OpenVLA models with various loss functions.
Supports multiple loss types for action token prediction:

    1. hard_ce: Standard hard cross-entropy loss (same as finetune.py)
       Only the correct bin gets weight 1, all others get 0.

    2. soft_ce (default): Gaussian soft target distribution centered on the correct bin,
       allowing the model to learn that nearby bins represent similar actions.

    3. wasserstein: 1D Wasserstein distance (Earth Mover's Distance) using closed-form
       CDF solution. Considers the geometry of the action space.
       Supports soft target (sigma > 0) for combining Wasserstein geometry with soft tolerance.

    4. sinkhorn: Entropy-regularized optimal transport distance. More general but
       computationally heavier. Also supports soft target (sigma > 0).

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

    - [Hard CE Loss]: torchrun ... vla-scripts/finetune_soft_target.py --loss_type hard_ce ...
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
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.training import compute_action_loss
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    # Options: "hard_ce", "soft_ce", "wasserstein", "sinkhorn", "expected_value"
    loss_type: str = "soft_ce"

    # Soft Target Parameters (used by soft_ce, wasserstein, sinkhorn)
    # sigma=1.0: ~68% weight within +/- 1 bin, sigma=2.0: +/- 2 bins
    # For wasserstein/sinkhorn: set to 0 for hard target
    soft_target_sigma: float = 2.0

    # Sinkhorn Parameters (only used when loss_type="sinkhorn")
    sinkhorn_epsilon: float = 0.1  # Entropy regularization
    sinkhorn_iters: int = 50

    # Wasserstein/Sinkhorn Scale (to match CE loss magnitude)
    # CE loss is typically 1-5, wasserstein is ~0.01-0.3, so scale ~10-20 helps
    ot_loss_scale: float = 10.0

    # Expected Value Loss Parameters (only used when loss_type="expected_value")
    # Regression loss on soft argmax: pred = sum(softmax * bin_centers)
    # Options: "l1", "l2", "smooth_l1"
    regression_loss_fn: str = "l1"

    # Gradient Clipping (0 = disabled, recommended: 1.0 for wasserstein/sinkhorn)
    grad_clip_norm: float = 1.0

    # Accuracy Margin Parameter (0 = strict, 3 = +/- 3 bins tolerance)
    action_accuracy_margin: int = 0

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
    if cfg.loss_type == "hard_ce":
        loss_info = "Hard CE Loss (standard cross-entropy)"
    elif cfg.loss_type == "soft_ce":
        loss_info = f"Soft CE Loss (sigma={cfg.soft_target_sigma})"
    elif cfg.loss_type == "wasserstein":
        sigma_str = f", sigma={cfg.soft_target_sigma}" if cfg.soft_target_sigma > 0 else ""
        loss_info = f"Wasserstein-1D Loss (scale={cfg.ot_loss_scale}{sigma_str})"
    elif cfg.loss_type == "sinkhorn":
        sigma_str = f", sigma={cfg.soft_target_sigma}" if cfg.soft_target_sigma > 0 else ""
        loss_info = f"Sinkhorn Loss (eps={cfg.sinkhorn_epsilon}, iters={cfg.sinkhorn_iters}, scale={cfg.ot_loss_scale}{sigma_str})"
    elif cfg.loss_type == "expected_value":
        loss_info = f"Expected Value Regression Loss ({cfg.regression_loss_fn})"
    else:
        loss_info = f"Unknown Loss ({cfg.loss_type})"
    grad_clip_info = f", grad_clip={cfg.grad_clip_norm}" if cfg.grad_clip_norm > 0 else ""
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}` with {loss_info}{grad_clip_info}")

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
    if cfg.loss_type == "hard_ce":
        exp_id += "+hard_ce"
    elif cfg.loss_type == "soft_ce":
        exp_id += f"+soft_ce-sigma{cfg.soft_target_sigma}"
    elif cfg.loss_type == "wasserstein":
        exp_id += f"+wasserstein-scale{cfg.ot_loss_scale}"
        if cfg.soft_target_sigma > 0:
            exp_id += f"-sigma{cfg.soft_target_sigma}"
    elif cfg.loss_type == "sinkhorn":
        exp_id += f"+sinkhorn-eps{cfg.sinkhorn_epsilon}-scale{cfg.ot_loss_scale}"
        if cfg.soft_target_sigma > 0:
            exp_id += f"-sigma{cfg.soft_target_sigma}"
    elif cfg.loss_type == "expected_value":
        exp_id += f"+expected_value-{cfg.regression_loss_fn}"
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

                # Compute action loss (soft_ce, wasserstein, sinkhorn, or expected_value)
                # Get continuous actions for expected_value loss (if available)
                continuous_actions = batch.get("continuous_actions")
                if continuous_actions is not None:
                    continuous_actions = continuous_actions.to(device_id)

                loss, _num_action_tokens = compute_action_loss(
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
                    ot_loss_scale=cfg.ot_loss_scale,
                    regression_loss_fn=cfg.regression_loss_fn,
                    continuous_actions=continuous_actions,
                )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, num_image_patches:-1]
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
                # Gradient clipping for training stability (especially useful for wasserstein/sinkhorn)
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint
            if (
                (batch_idx + 1) % cfg.grad_accumulation_steps == 0
                and gradient_step_idx > 0
                and gradient_step_idx % cfg.save_steps == 0
            ):
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
