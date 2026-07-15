"""
Unified seed management for reproducible experiments.

All randomness is derived deterministically from one master seed. Server and
serverless experiments use the same rules so corresponding runs share model
initialization and data partitions.

Seed derivation rules:
- Initial weights: MASTER_SEED
- Data partition: MASTER_SEED + 101
- Training DataLoader: MASTER_SEED + client_id * 1000 + 1
- Evaluation DataLoader: MASTER_SEED + client_id * 1000 + 2
- Client sampling: MASTER_SEED + round * 100 + 3
- DP noise: MASTER_SEED + client_id * 1000 + round
- Topology generation: MASTER_SEED + 404
"""

import os
import torch
import numpy as np
import random
from typing import Optional


class SeedManager:
    """Derive reproducible seeds for every source of randomness."""

    # Fixed offsets keep independent random streams reproducible.
    DATA_SPLIT_OFFSET = 101
    TRAIN_LOADER_BASE = 1
    EVAL_LOADER_BASE = 2
    CLIENT_SAMPLE_BASE = 3
    TOPOLOGY_OFFSET = 404
    CLIENT_MULTIPLIER = 1000
    ROUND_MULTIPLIER = 100

    def __init__(self, master_seed: Optional[int] = None):
        """
        Initialize the seed manager.

        Args:
            master_seed: Master seed. If omitted, read FL_SEED or use 2025.
        """
        if master_seed is None:
            master_seed = int(os.environ.get("FL_SEED", "2025"))

        self.master_seed = master_seed
        print(f"[SEED MANAGER] Initialized with MASTER_SEED={self.master_seed}")

    def get_master_seed(self) -> int:
        """Return the master seed."""
        return self.master_seed

    def get_init_weights_seed(self) -> int:
        """Return the initial-weight seed."""
        return self.master_seed

    def get_data_split_seed(self) -> int:
        """Return the data-partition seed."""
        seed = self.master_seed + self.DATA_SPLIT_OFFSET
        return seed

    def get_train_loader_seed(self, client_id: int, round_num: int = 0) -> int:
        """
        Return a round-specific training DataLoader seed.

        Args:
            client_id: Client identifier.
            round_num: Zero-based round number.

        Note:
            Each round receives a different, reproducible batch order.
        """
        seed = self.master_seed + client_id * self.CLIENT_MULTIPLIER + round_num
        return seed

    def get_eval_loader_seed(self, client_id: int) -> int:
        """
        Return the evaluation DataLoader seed.

        Args:
            client_id: Client identifier.
        """
        seed = self.master_seed + client_id * self.CLIENT_MULTIPLIER + self.EVAL_LOADER_BASE
        return seed

    def get_client_sample_seed(self, round_num: int) -> int:
        """
        Return the client-sampling seed.

        Args:
            round_num: Round number.
        """
        seed = self.master_seed + round_num * self.ROUND_MULTIPLIER + self.CLIENT_SAMPLE_BASE
        return seed

    def get_dp_noise_seed(self, client_id: int, round_num: int) -> int:
        """
        Return a client- and round-specific DP noise seed.

        Args:
            client_id: Client identifier.
            round_num: Round number.
        """
        seed = self.master_seed + client_id * self.CLIENT_MULTIPLIER + round_num
        return seed

    def get_topology_seed(self) -> int:
        """Return the topology-generation seed."""
        seed = self.master_seed + self.TOPOLOGY_OFFSET
        return seed

    def set_global_seed(self, seed: Optional[int] = None):
        """
        Set global random seeds at experiment startup.

        Args:
            seed: Seed value. If omitted, use the master seed.
        """
        if seed is None:
            seed = self.master_seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Configure deterministic backend behavior.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Strong deterministic algorithms may reduce performance.
        # torch.use_deterministic_algorithms(True)

        print(f"[SEED MANAGER] Global seed set to {seed}")
        print(f"[SEED MANAGER] CUDNN deterministic mode: ON")

    def print_seed_summary(self):
        """Print a summary of derived seeds."""
        print("=" * 80)
        print("[SEED SUMMARY] Seed Configuration")
        print("=" * 80)
        print(f"  MASTER_SEED: {self.master_seed}")
        print(f"  Initial Weights Seed: {self.get_init_weights_seed()}")
        print(f"  Data Split Seed: {self.get_data_split_seed()}")
        print(f"  Topology Seed: {self.get_topology_seed()}")
        print(f"\nDerived Seed Formulas:")
        print(f"  Train Loader (client i, round r): MASTER + i*{self.CLIENT_MULTIPLIER} + r")
        print(f"  Eval Loader (client i): MASTER + i*{self.CLIENT_MULTIPLIER} + {self.EVAL_LOADER_BASE}")
        print(f"  Client Sample (round r): MASTER + r*{self.ROUND_MULTIPLIER} + {self.CLIENT_SAMPLE_BASE}")
        print(f"  DP Noise (client i, round r): MASTER + i*{self.CLIENT_MULTIPLIER} + r")
        print("=" * 80)


# Shared process-local seed manager.
_global_seed_manager: Optional[SeedManager] = None


def get_seed_manager(master_seed: Optional[int] = None) -> SeedManager:
    """
    Return the shared seed manager.

    Args:
        master_seed: Master seed used only on first initialization.

    Returns:
        SeedManager instance.
    """
    global _global_seed_manager

    if _global_seed_manager is None:
        _global_seed_manager = SeedManager(master_seed)

    return _global_seed_manager


def reset_seed_manager(master_seed: Optional[int] = None):
    """
    Reset the shared seed manager.

    Args:
        master_seed: New master seed.

    Returns:
        SeedManager: New seed manager instance.
    """
    global _global_seed_manager
    _global_seed_manager = SeedManager(master_seed)
    return _global_seed_manager


def set_deterministic_mode():
    """Configure deterministic PyTorch behavior."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[DETERMINISTIC] CUDNN deterministic mode enabled")
    print("[DETERMINISTIC] CUDNN benchmark mode disabled")


# Convenience accessors.
def get_init_seed() -> int:
    """Return the initial-weight seed."""
    return get_seed_manager().get_init_weights_seed()


def get_split_seed() -> int:
    """Return the data-partition seed."""
    return get_seed_manager().get_data_split_seed()


def get_train_seed(client_id: int, round_num: int = 0) -> int:
    """Return a round-specific training-loader seed."""
    return get_seed_manager().get_train_loader_seed(client_id, round_num)


def get_eval_seed(client_id: int) -> int:
    """Return the evaluation-loader seed."""
    return get_seed_manager().get_eval_loader_seed(client_id)


def get_sample_seed(round_num: int) -> int:
    """Return the client-sampling seed."""
    return get_seed_manager().get_client_sample_seed(round_num)


def get_noise_seed(client_id: int, round_num: int) -> int:
    """Return the DP noise seed."""
    return get_seed_manager().get_dp_noise_seed(client_id, round_num)


def get_topo_seed() -> int:
    """Return the topology seed."""
    return get_seed_manager().get_topology_seed()


if __name__ == "__main__":
    # Lightweight deterministic self-check.
    print("\n" + "="*80)
    print("SEED MANAGER TEST")
    print("="*80)

    # Default seed.
    print("\n[TEST 1] Default Master Seed (2025)")
    sm1 = SeedManager()
    sm1.print_seed_summary()

    print(f"\nExample derived seeds:")
    print(f"  Client 0, Round 1 - Train: {sm1.get_train_loader_seed(0, 1)}, DP Noise: {sm1.get_dp_noise_seed(0, 1)}")
    print(f"  Client 5, Round 10 - Train: {sm1.get_train_loader_seed(5, 10)}, DP Noise: {sm1.get_dp_noise_seed(5, 10)}")

    # Custom seed.
    print("\n[TEST 2] Custom Master Seed (42)")
    sm2 = SeedManager(master_seed=42)
    sm2.print_seed_summary()

    print(f"\nExample derived seeds:")
    print(f"  Client 0, Round 1 - Train: {sm2.get_train_loader_seed(0, 1)}, DP Noise: {sm2.get_dp_noise_seed(0, 1)}")
    print(f"  Client 5, Round 10 - Train: {sm2.get_train_loader_seed(5, 10)}, DP Noise: {sm2.get_dp_noise_seed(5, 10)}")

    # Consistency check.
    print("\n[TEST 3] Consistency Verification")
    print("Same master seed should generate same derived seeds:")
    sm3a = SeedManager(master_seed=100)
    sm3b = SeedManager(master_seed=100)

    assert sm3a.get_data_split_seed() == sm3b.get_data_split_seed()
    assert sm3a.get_train_loader_seed(0) == sm3b.get_train_loader_seed(0)
    assert sm3a.get_dp_noise_seed(5, 10) == sm3b.get_dp_noise_seed(5, 10)
    print(" All consistency checks passed!")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
