"""
loss.py

Loss functions for VLA action prediction training.
Supports multiple loss types:
    - hard_ce: Standard hard cross-entropy loss
    - soft_ce: Gaussian soft target cross-entropy loss
    - wasserstein: 1D Wasserstein distance (Earth Mover's Distance)
    - sinkhorn: Entropy-regularized optimal transport distance
"""

import torch
import torch.nn.functional as F


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

    # Create bin indices [0, 1, ..., num_classes-1]
    bin_indices = torch.arange(num_classes, device=device, dtype=torch.float32)  # [num_classes]

    # Expand for broadcasting: target_indices [N, 1], bin_indices [1, num_classes]
    target_indices = target_indices.float().unsqueeze(1)  # [N, 1]
    bin_indices = bin_indices.unsqueeze(0)  # [1, num_classes]

    # Compute Gaussian weights: exp(-0.5 * ((bin - target) / sigma)^2)
    distances = (bin_indices - target_indices) / sigma  # [N, num_classes]
    soft_targets = torch.exp(-0.5 * distances**2)  # [N, num_classes]

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


def hard_cross_entropy_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute standard (hard) cross-entropy loss.

    This is the same loss used in the original finetune.py - only the correct
    bin gets probability 1, all others get 0.

    Args:
        logits: Raw logits of shape [N, num_classes]
        target_indices: Target class indices of shape [N]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Cross-entropy loss
    """
    return F.cross_entropy(logits, target_indices, reduction=reduction)


def wasserstein_1d_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    n_bins: int = 256,
    sigma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance) between predicted distribution and target.

    For 1D distributions, Wasserstein-1 has a closed-form solution using CDFs:
        W_1(P, Q) = sum(|CDF_P - CDF_Q|)

    This loss considers the geometry of the action space - nearby bins are treated as more similar.
    When sigma > 0, uses Gaussian soft target instead of one-hot, combining benefits of both
    Wasserstein geometry and soft target tolerance.

    Args:
        logits: Raw logits of shape [N, n_bins]
        target_indices: Target bin indices of shape [N]
        n_bins: Number of action bins
        sigma: Gaussian sigma for soft targets. If 0, uses hard (one-hot) target.
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Wasserstein-1 distance loss
    """
    # Convert logits to probabilities with numerical stability
    # Subtract max for numerical stability before softmax
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values
    pred_probs = F.softmax(logits_stable, dim=-1)  # [N, n_bins]

    # Add small epsilon and renormalize to avoid numerical issues in CDF
    pred_probs = pred_probs + 1e-8
    pred_probs = pred_probs / pred_probs.sum(dim=-1, keepdim=True)

    # Create target distribution (soft or hard)
    if sigma > 0:
        # Gaussian soft target
        target_probs = create_gaussian_soft_target(
            target_indices,
            num_classes=n_bins,
            sigma=sigma,
            device=logits.device,
        )
    else:
        # One-hot hard target
        target_probs = torch.zeros_like(pred_probs)
        target_probs.scatter_(1, target_indices.unsqueeze(1), 1.0)

    # Compute CDFs (cumulative distribution functions)
    pred_cdf = torch.cumsum(pred_probs, dim=-1)
    target_cdf = torch.cumsum(target_probs, dim=-1)

    # Clamp CDFs to [0, 1] for numerical stability
    pred_cdf = torch.clamp(pred_cdf, 0.0, 1.0)
    target_cdf = torch.clamp(target_cdf, 0.0, 1.0)

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
    sigma: float = 0.0,
    epsilon: float = 0.1,
    n_iters: int = 50,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Sinkhorn distance (entropy-regularized Wasserstein) between predicted distribution and target.

    Sinkhorn provides a differentiable approximation to optimal transport distance.
    For 1D case, this is more expensive than closed-form Wasserstein but more general.
    When sigma > 0, uses Gaussian soft target instead of one-hot.

    Args:
        logits: Raw logits of shape [N, n_bins]
        target_indices: Target bin indices of shape [N]
        n_bins: Number of action bins
        sigma: Gaussian sigma for soft targets. If 0, uses hard (one-hot) target.
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

    # Create target distribution (soft or hard)
    if sigma > 0:
        # Gaussian soft target
        target_probs = create_gaussian_soft_target(
            target_indices,
            num_classes=n_bins,
            sigma=sigma,
            device=device,
        )
    else:
        # One-hot hard target
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


def compute_action_loss(
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
    ot_loss_scale: float = 10.0,
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
            - "hard_ce": Standard cross-entropy (same as finetune.py)
            - "soft_ce": Soft cross-entropy with Gaussian targets (default)
            - "wasserstein": 1D Wasserstein distance (Earth Mover's Distance)
            - "sinkhorn": Sinkhorn distance (entropy-regularized OT)
        sigma: Gaussian sigma for soft targets (used by soft_ce, wasserstein, sinkhorn)
        sinkhorn_epsilon: Entropy regularization for Sinkhorn (only used when loss_type="sinkhorn")
        sinkhorn_iters: Number of Sinkhorn iterations (only used when loss_type="sinkhorn")
        ot_loss_scale: Scale factor for wasserstein/sinkhorn losses (to match CE magnitude)

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
    # This means: token (vocab_size - 1) -> bin 0, token (vocab_size - n_bins) -> bin (n_bins - 1)
    action_bin_indices = vocab_size - action_labels - 1

    # Clamp to valid range (handles edge cases)
    action_bin_indices = torch.clamp(action_bin_indices, 0, n_bins - 1)

    # Extract only action token logits from positions (vocab_size - n_bins) to (vocab_size - 1)
    # Note: We use tokenizer's vocab_size, not logits_vocab_size (which may be padded)
    # Action tokens are at vocab_size - n_bins (bin 255) to vocab_size - 1 (bin 0)
    # We flip to get logits ordered by bin index: logit[0] = bin 0, logit[255] = bin 255
    action_token_logits = action_logits[:, vocab_size - n_bins : vocab_size].flip(
        dims=[-1]
    )  # [num_action_tokens, n_bins]

    # Compute loss based on loss_type
    if loss_type == "hard_ce":
        # Standard hard cross-entropy (same as finetune.py)
        loss = hard_cross_entropy_loss(action_token_logits, action_bin_indices, reduction="mean")

    elif loss_type == "soft_ce":
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
        # Uses soft target if sigma > 0, combining Wasserstein geometry with soft target tolerance
        # Scale to match CE loss magnitude for stable training
        loss = (
            wasserstein_1d_loss(
                action_token_logits,
                action_bin_indices,
                n_bins=n_bins,
                sigma=sigma,
                reduction="mean",
            )
            * ot_loss_scale
        )

    elif loss_type == "sinkhorn":
        # Sinkhorn distance (entropy-regularized optimal transport)
        # Uses soft target if sigma > 0
        # Scale to match CE loss magnitude for stable training
        loss = (
            sinkhorn_loss(
                action_token_logits,
                action_bin_indices,
                n_bins=n_bins,
                sigma=sigma,
                epsilon=sinkhorn_epsilon,
                n_iters=sinkhorn_iters,
                reduction="mean",
            )
            * ot_loss_scale
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'hard_ce', 'soft_ce', 'wasserstein', 'sinkhorn'.")

    return loss, action_mask.sum().item()
