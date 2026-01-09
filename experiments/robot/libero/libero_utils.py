"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import io

import imageio
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def create_action_prob_plot(all_action_probs, selected_actions=None, step_idx=0, figsize=(8, 12), dpi=150):
    """
    Creates a HIGH RESOLUTION figure showing probability distributions for all action dimensions.

    Args:
        all_action_probs: List of 7 numpy arrays, each of shape (256,) containing probabilities
        selected_actions: Optional list of 7 selected action values (normalized, -1 to 1)
        step_idx: Current step index for title
        figsize: Figure size in inches
        dpi: DPI for high quality output

    Returns:
        numpy array of the rendered figure (RGB image) at high resolution
    """
    from PIL import Image

    if not all_action_probs or len(all_action_probs) == 0:
        # Return a blank image if no data
        blank = np.ones((600, 400, 3), dtype=np.uint8) * 255
        return blank

    n_dims = len(all_action_probs)
    n_bins = len(all_action_probs[0])

    # Create bin centers for x-axis
    bin_centers = np.linspace(-1, 1, n_bins)

    action_dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    fig, axes = plt.subplots(n_dims, 1, figsize=figsize)
    if n_dims == 1:
        axes = [axes]

    for i, (probs, ax) in enumerate(zip(all_action_probs, axes)):
        # Use fill_between for smooth visualization
        ax.fill_between(bin_centers, probs, alpha=0.7, color='steelblue')
        ax.plot(bin_centers, probs, color='darkblue', linewidth=1)

        # Mark the selected action with a red vertical line
        if selected_actions is not None and i < len(selected_actions):
            selected_val = float(selected_actions[i])
            ax.axvline(x=selected_val, color='red', linestyle='-', linewidth=2.5,
                      label=f'Selected: {selected_val:.3f}')
            ax.legend(loc='upper right', fontsize=10)

        dim_name = action_dim_names[i] if i < len(action_dim_names) else f"dim{i}"
        ax.set_ylabel(dim_name, fontsize=12, fontweight='bold')
        ax.set_xlim(-1.05, 1.05)

        # Set y-axis limit with some padding
        max_prob = float(probs.max()) if probs.max() > 0 else 0.1
        ax.set_ylim(0, max_prob * 1.3)

        # Only show x-axis label for the bottom plot
        if i == n_dims - 1:
            ax.set_xlabel('Action Value (normalized)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Step {step_idx}: Action Probabilities', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Convert figure to numpy array at HIGH DPI
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)

    # Read image from buffer (keep at high resolution)
    img = Image.open(buf)
    img_array = np.array(img.convert('RGB'))

    plt.close(fig)
    buf.close()

    return img_array


def combine_image_with_prob_plot(robot_image, prob_plot_image):
    """
    Combines robot observation image with probability plot image side by side at HIGH RESOLUTION.
    The prob_plot height determines the combined height.

    Args:
        robot_image: numpy array of robot observation (H, W, 3)
        prob_plot_image: numpy array of probability plot (H, W, 3) - high resolution

    Returns:
        Combined numpy array at high resolution
    """
    from PIL import Image

    # Convert to PIL Images
    robot_pil = Image.fromarray(robot_image)
    prob_pil = Image.fromarray(prob_plot_image)

    # Use prob_plot height as the target height (high res)
    target_height = prob_plot_image.shape[0]

    # Resize robot image to match height (keeping square aspect ratio)
    robot_pil = robot_pil.resize((target_height, target_height), Image.LANCZOS)

    # Combined width = robot (square) + prob plot
    combined_width = target_height + prob_plot_image.shape[1]

    # Combine horizontally at high resolution
    combined = Image.new('RGB', (combined_width, target_height), color=(255, 255, 255))
    combined.paste(robot_pil, (0, 0))
    combined.paste(prob_pil, (target_height, 0))

    return np.array(combined)


def save_rollout_video_with_probs(rollout_images, prob_data_list, idx, success, task_description,
                                  log_file=None, video_height=480):
    """
    Saves an MP4 replay with probability distribution plots.
    Plots are generated at HIGH RESOLUTION, then resized for video output.

    Args:
        rollout_images: List of robot observation images
        prob_data_list: List of dicts with 'all_action_probs' and 'normalized_actions' keys
                       (can have None entries for steps without prob logging)
        idx: Episode index
        success: Whether episode was successful
        task_description: Task description string
        log_file: Optional log file handle
        video_height: Output video height (width is auto-calculated)

    Returns:
        Path to saved MP4 file
    """
    from PIL import Image

    if not rollout_images:
        print("Warning: No rollout images to save")
        return None

    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}--with_probs.mp4"

    try:
        last_prob_plot = None  # Cache last probability plot (high res)
        last_combined_highres = None  # Cache last combined frame (high res)
        video_width = None  # Will be determined from first frame

        # Pad prob_data_list if shorter than rollout_images
        prob_data_list = list(prob_data_list)  # Make a copy
        while len(prob_data_list) < len(rollout_images):
            prob_data_list.append(None)

        frames = []

        for step_idx, (img, prob_data) in enumerate(zip(rollout_images, prob_data_list)):
            try:
                if prob_data is not None and 'all_action_probs' in prob_data:
                    # Create HIGH RESOLUTION probability plot
                    prob_plot = create_action_prob_plot(
                        prob_data['all_action_probs'],
                        selected_actions=prob_data.get('normalized_actions'),
                        step_idx=step_idx,
                        figsize=(8, 12),  # Large figure
                        dpi=150  # High DPI
                    )
                    last_prob_plot = prob_plot

                if last_prob_plot is not None:
                    # Combine at HIGH RESOLUTION
                    combined_highres = combine_image_with_prob_plot(img, last_prob_plot)
                    last_combined_highres = combined_highres
                else:
                    # No probability data yet, just use robot image (upscaled)
                    robot_pil = Image.fromarray(img)
                    # Create a placeholder combined image
                    placeholder_height = 600
                    robot_pil = robot_pil.resize((placeholder_height, placeholder_height), Image.LANCZOS)
                    combined_highres = np.array(robot_pil)
                    last_combined_highres = combined_highres

                # Determine video dimensions from first valid frame
                if video_width is None:
                    aspect_ratio = combined_highres.shape[1] / combined_highres.shape[0]
                    video_width = int(video_height * aspect_ratio)
                    # Make sure width is even (required for some codecs)
                    video_width = video_width + (video_width % 2)

                # Resize HIGH RES frame to video size
                frame_pil = Image.fromarray(combined_highres)
                frame_pil = frame_pil.resize((video_width, video_height), Image.LANCZOS)
                frame = np.array(frame_pil)

                frames.append(frame)

            except Exception as e:
                print(f"Warning: Error processing frame {step_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not frames:
            print("Warning: No frames were generated")
            return None

        # Write video
        video_writer = imageio.get_writer(mp4_path, fps=10, codec='libx264', pixelformat='yuv420p')
        for frame in frames:
            video_writer.append_data(frame)
        video_writer.close()

        print(f"Saved rollout MP4 with probs ({video_width}x{video_height}) at path {mp4_path}")
        if log_file is not None:
            log_file.write(f"Saved rollout MP4 with probs at path {mp4_path}\n")
        return mp4_path

    except Exception as e:
        print(f"Error saving video with probs: {e}")
        import traceback
        traceback.print_exc()
        if log_file is not None:
            log_file.write(f"Error saving video with probs: {e}\n")
        return None


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
