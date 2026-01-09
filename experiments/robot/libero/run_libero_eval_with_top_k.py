"""
run_libero_eval_with_top_k.py

Runs a model in a LIBERO simulation environment with top-k action token probability logging.

Usage:
    python experiments/robot/libero/run_libero_eval_with_top_k.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True \
        --top_k 10 \
        --log_probs_every_n_steps 1
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
from PIL import Image

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    save_rollout_video_with_probs,
)
from experiments.robot.openvla_utils import OPENVLA_V01_SYSTEM_PROMPT, crop_and_resize, get_processor
from experiments.robot.robot_utils import (
    ACTION_DIM,
    DATE_TIME,
    DEVICE,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# Try importing tensorflow for center crop
try:
    import tensorflow as tf
except ImportError:
    tf = None


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Top-k probability logging parameters
    #################################################################################################################
    top_k: int = 10                                  # Number of top tokens to log
    log_probs_every_n_steps: int = 1                 # Log probabilities every N steps (1 = every step)
    save_prob_distribution: bool = False             # Save full probability distribution to file

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def get_vla_action_with_probs(
    vla,
    processor,
    base_vla_name: str,
    obs: dict,
    task_label: str,
    unnorm_key: str,
    center_crop: bool = False,
    top_k: int = 10,
    step_idx: int = 0,
    log_file=None,
):
    """
    Generates an action with the VLA policy and logs top-k token probabilities.

    Returns:
        action: numpy array of shape (7,)
        prob_info: dict with probability information
    """
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action dimension
    action_dim = vla.get_action_dim(unnorm_key)

    # If the special empty token ('') does not already appear after the colon (':') token
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )
        inputs["input_ids"] = input_ids

    # Generate with output scores
    with torch.inference_mode():
        outputs = vla.generate(
            **inputs,
            max_new_tokens=action_dim,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Get generated token IDs and scores
    generated_ids = outputs.sequences[0, -action_dim:]
    scores = outputs.scores  # Tuple of length action_dim

    # Get vocab size and compute bin centers
    # Note: vla.vocab_size already has pad_to_multiple_of subtracted (see modeling_prismatic.py line 504)
    vocab_size = vla.vocab_size

    n_bins = 256
    action_token_start = vocab_size - n_bins
    bins = np.linspace(-1, 1, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Log top-k probabilities
    output_lines = []
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"[Step {step_idx}] Action Token Probabilities (top-{top_k})")
    output_lines.append(f"{'='*80}")

    all_action_probs = []
    normalized_actions = []

    action_dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    for dim_idx, (token_id, score) in enumerate(zip(generated_ids, scores)):
        # Convert logits to probabilities
        probs = torch.softmax(score[0], dim=-1)

        # Get top-k tokens and their probabilities
        top_probs, top_indices = torch.topk(probs, top_k)

        # Get the selected token's probability
        selected_prob = probs[token_id].item()

        # Decode action value
        discretized = vocab_size - token_id.item()
        action_idx = np.clip(discretized - 1, 0, len(bin_centers) - 1)
        action_value = bin_centers[action_idx]
        normalized_actions.append(action_value)

        dim_name = action_dim_names[dim_idx] if dim_idx < len(action_dim_names) else f"dim{dim_idx}"
        output_lines.append(
            f"\n  [{dim_name}] Selected: token={token_id.item()}, prob={selected_prob:.4f}, "
            f"action={action_value:+.4f}"
        )
        output_lines.append(f"  Top-{top_k}:")

        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            idx_val = idx.item()
            prob_val = prob.item()

            is_action_token = idx_val >= action_token_start
            if is_action_token:
                disc = vocab_size - idx_val
                act_idx = np.clip(disc - 1, 0, len(bin_centers) - 1)
                act_val = bin_centers[act_idx]
                marker = " <--" if idx_val == token_id.item() else ""
                output_lines.append(
                    f"    {i+1:2d}. Token {idx_val:5d}: prob={prob_val:.4f}, action={act_val:+.4f}{marker}"
                )
            else:
                marker = " <-- (non-action!)" if idx_val == token_id.item() else " (non-action)"
                output_lines.append(f"    {i+1:2d}. Token {idx_val:5d}: prob={prob_val:.4f}{marker}")

        # Store action token probabilities (exactly n_bins elements)
        # Note: probs may have padded vocab size, so we slice [action_token_start:vocab_size]
        # Flip to align with bin order: token 31744 (bin 255, action +1) -> index 255
        #                               token 31999 (bin 0, action -1) -> index 0
        action_probs = probs[action_token_start:vocab_size].cpu().numpy()[::-1]
        all_action_probs.append(action_probs)

    normalized_actions = np.array(normalized_actions)

    # Unnormalize actions
    action_norm_stats = vla.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    output_lines.append(f"\n  Normalized actions:   {normalized_actions}")
    output_lines.append(f"  Unnormalized actions: {actions}")

    # Print and log
    output_text = "\n".join(output_lines)
    print(output_text)
    if log_file:
        log_file.write(output_text + "\n")
        log_file.flush()

    prob_info = {
        "all_action_probs": all_action_probs,
        "generated_token_ids": generated_ids.cpu().numpy(),
        "normalized_actions": normalized_actions,
    }

    return actions, prob_info


def get_action_with_probs(cfg, model, obs, task_label, processor, step_idx, log_file):
    """Queries the model to get an action with probability logging."""
    if cfg.model_family == "openvla":
        action, prob_info = get_vla_action_with_probs(
            model,
            processor,
            cfg.pretrained_checkpoint,
            obs,
            task_label,
            cfg.unnorm_key,
            center_crop=cfg.center_crop,
            top_k=cfg.top_k,
            step_idx=step_idx,
            log_file=log_file,
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action, prob_info


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-TOPK-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Top-k logging: k={cfg.top_k}, every {cfg.log_probs_every_n_steps} steps")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Top-k logging: k={cfg.top_k}, every {cfg.log_probs_every_n_steps} steps\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            prob_data_list = []  # Store probability data for video with probs
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            else:
                max_steps = 300

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            step_count = 0  # Counter for actual action steps (excluding wait steps)

            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    if t < cfg.num_steps_wait:
                        obs, _reward, _done, _info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action (with probability logging)
                    should_log = step_count % cfg.log_probs_every_n_steps == 0

                    if should_log:
                        action, prob_info = get_action_with_probs(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor,
                            step_idx=step_count,
                            log_file=log_file,
                        )
                        # Store probability data for video
                        prob_data_list.append(prob_info)
                    else:
                        # Use regular action prediction without logging
                        from experiments.robot.robot_utils import get_action

                        action = get_action(cfg, model, observation, task_description, processor=processor)
                        prob_data_list.append(None)  # No prob data for this step

                    step_count += 1

                    # Normalize gripper action [0,1] -> [-1,+1]
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] Flip gripper action sign
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, _reward, done, _info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode (original)
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Save a replay video with probability plots
            save_rollout_video_with_probs(
                replay_images,
                prob_data_list,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
