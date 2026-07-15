# Privacy Accountant
"""
Privacy Accountant
Tracks cumulative privacy loss in federated learning
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PrivacyEvent:
    round_num: int
    client_id: Optional[int]
    mechanism: str
    epsilon: float
    delta: float = 0.0
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class PrivacyAccountant:
    def __init__(self, save_path: Optional[str] = None, target_delta: float = 1e-5):
        self.events: List[PrivacyEvent] = []
        self.save_path = save_path
        self.target_delta = target_delta
        self.steps_taken = 0

        # Laplace mechanism: linear accumulation
        self.laplace_epsilon_total = 0.0
        self.laplace_client_epsilon: Dict[int, float] = {}

        # Client-level Gaussian: linear epsilon accumulation
        self.client_gaussian_epsilon_total = 0.0
        self.client_gaussian_client_epsilon: Dict[int, float] = {}

    def add_laplace_event(self, round_num: int, epsilon: float, client_id: Optional[int] = None):
        event = PrivacyEvent(
            round_num=round_num,
            client_id=client_id,
            mechanism="laplace",
            epsilon=epsilon,
            delta=0.0
        )
        self.events.append(event)

        self.laplace_epsilon_total += epsilon

        if client_id is not None:
            self.laplace_client_epsilon[client_id] = self.laplace_client_epsilon.get(client_id, 0.0) + epsilon

        if self.save_path:
            self.save_to_file()

    def add_client_level_gaussian_event(self, round_num: int, epsilon: float, delta: float, client_id: Optional[int] = None):
        event = PrivacyEvent(
            round_num=round_num,
            client_id=client_id,
            mechanism="client_gaussian",
            epsilon=float(epsilon),
            delta=float(delta)
        )
        self.events.append(event)

        self.client_gaussian_epsilon_total += float(epsilon)
        if client_id is not None:
            self.client_gaussian_client_epsilon[client_id] = self.client_gaussian_client_epsilon.get(client_id, 0.0) + float(epsilon)

        if self.save_path:
            self.save_to_file()

    def add_gaussian_event(self, noise_multiplier: float, sampling_probability: float, steps: int = 1, **kwargs):
        """
        Gaussian event logging method compatible with the Opacus RDPAccountant interface.
        Converts noise_multiplier and sampling_probability into the equivalent epsilon value.
        """
        # Estimate the equivalent epsilon using the RDP formula
        if noise_multiplier > 0 and sampling_probability > 0:
            # epsilon approximately (sampling_probability * steps) / noise_multiplier^2 * log(1/delta)
            import math
            delta = kwargs.get('delta', self.target_delta)
            epsilon_per_step = (sampling_probability / (noise_multiplier ** 2)) * math.log(1/delta)
            total_epsilon = epsilon_per_step * steps
        else:
            total_epsilon = 0.0
            delta = self.target_delta

        round_num = kwargs.get('round_num', self.steps_taken + 1)
        client_id = kwargs.get('client_id', None)

        self.add_client_level_gaussian_event(
            round_num=round_num,
            epsilon=total_epsilon,
            delta=delta,
            client_id=client_id
        )

        self.steps_taken += steps

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Get the epsilon value compatible with the Opacus RDPAccountant interface.
        """
        eps, _ = self.get_privacy_spent()
        return eps

    def get_privacy_spent(self) -> Tuple[float, float]:
        laplace_eps = self.laplace_epsilon_total
        client_gauss_eps = self.client_gaussian_epsilon_total

        total_eps = laplace_eps + client_gauss_eps
        total_delta = self.target_delta if client_gauss_eps > 0 else 0.0

        return total_eps, total_delta

    def update_laplace(self, round_num: int, epsilon: float, client_id: Optional[int] = None):
        self.add_laplace_event(round_num, epsilon, client_id)

    def get_client_privacy_spent(self, client_id: int) -> Tuple[float, float]:
        laplace_eps = self.laplace_client_epsilon.get(client_id, 0.0)
        client_gauss_eps = self.client_gaussian_client_epsilon.get(client_id, 0.0)

        total_eps = laplace_eps + client_gauss_eps
        total_delta = self.target_delta if client_gauss_eps > 0 else 0.0

        return total_eps, total_delta

    def get_summary(self) -> str:
        eps, delta = self.get_privacy_spent()
        if delta > 0:
            return f"Total Privacy: (epsilon={eps:.4f}, delta={delta:.6f})"
        else:
            return f"Total Privacy: epsilon={eps:.4f}"

    def print_status(self, round_num: Optional[int] = None):
        summary = self.get_summary()
        if round_num is not None:
            print(f"[Privacy Accountant] Round {round_num}: {summary}")
        else:
            print(f"[Privacy Accountant] {summary}")

    def save_to_file(self):
        return

    def reset(self):
        self.events.clear()
        self.laplace_epsilon_total = 0.0
        self.laplace_client_epsilon.clear()
        self.client_gaussian_epsilon_total = 0.0
        self.client_gaussian_client_epsilon.clear()

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        eps, _ = self.get_privacy_spent()
        return eps


class LaplacePrivacyAccountant(PrivacyAccountant):
    def __init__(self, epsilon_per_round: float, save_path: Optional[str] = None):
        super().__init__(save_path)
        self.epsilon_per_round = epsilon_per_round
        self.rounds_completed = 0
        self.steps_taken = 0

    def step(self, round_num: Optional[int] = None):
        if round_num is None:
            round_num = self.rounds_completed

        self.add_laplace_event(round_num, self.epsilon_per_round)
        self.rounds_completed += 1
        self.steps_taken += 1

        self.print_status(round_num)

    def get_expected_epsilon(self, num_rounds: int) -> float:
        return self.epsilon_per_round * num_rounds

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        eps, _ = self.get_privacy_spent()
        return eps


if __name__ == "__main__":
    print("=== Privacy Accountant Test ===")

    accountant = PrivacyAccountant()

    for round_num in range(1, 6):
        accountant.update_laplace(round_num, 0.1, client_id=round_num % 3)
        accountant.print_status(round_num)

    print(f"\nFinal privacy consumption: {accountant.get_summary()}")
    print(f"Client 0 privacy consumption: epsilon={accountant.get_client_privacy_spent(0)[0]:.4f}")

    print("\n=== Laplace Specialized Accountant Test ===")
    lap_accountant = LaplacePrivacyAccountant(epsilon_per_round=0.1)

    for i in range(3):
        lap_accountant.step()

    print(f"Expected total epsilon after 50 rounds: {lap_accountant.get_expected_epsilon(50):.2f}")


# ---- Client-level Gaussian RDP Accountant ----
from typing import Iterable

class ClientGaussianRDPAccountant:
    """
    Client-level DP (per round, client sampling q, Gaussian noise with sigma).
    Accumulates RDP across rounds; converts to (epsilon, delta) on demand.

    Prefer to use Opacus/TF-Privacy's subsampled Gaussian RDP if available.
    Fallback to non-subsampled RDP as a conservative upper bound.
    """
    def __init__(self, orders: Optional[Iterable[float]] = None):
        # Typical orders; can be tuned if needed
        self.orders: List[float] = list(orders) if orders is not None else \
            [1.25, 1.5, 2, 2.5, 3, 4, 8, 16, 32, 64, 128, 256]
        self.rdp_cumulative = [0.0] * len(self.orders)

        # Try import RDP calculators (Opacus -> TF-Privacy)
        self._rdp_available = False
        self._use_tf_privacy = False
        self._use_opacus = False

        try:
            # TF-Privacy style API
            from privacy.analysis.rdp_accountant import compute_rdp as tfp_compute_rdp  # type: ignore
            from privacy.analysis.rdp_accountant import get_privacy_spent as tfp_get_privacy_spent  # type: ignore
            self._tfp_compute_rdp = tfp_compute_rdp
            self._tfp_get_privacy_spent = tfp_get_privacy_spent
            self._rdp_available = True
            self._use_tf_privacy = True
            print("[ClientGaussianRDPAccountant] Using TensorFlow Privacy RDP")
        except Exception:
            try:
                # Opacus >= 1.4 has analysis.rdp with subsampling
                from opacus.accountants.analysis.rdp import compute_rdp as opacus_compute_rdp  # type: ignore
                from opacus.accountants.analysis.rdp import get_privacy_spent as opacus_get_privacy_spent  # type: ignore
                self._opacus_compute_rdp = opacus_compute_rdp
                self._opacus_get_privacy_spent = opacus_get_privacy_spent
                self._rdp_available = True
                self._use_opacus = True
                print("[ClientGaussianRDPAccountant] Using Opacus RDP")
            except Exception:
                self._rdp_available = False
                print("[ClientGaussianRDPAccountant] Using fallback RDP (conservative upper bound)")

    def add_round(self, q: float, sigma: float, steps: int = 1):
        """
        Add one or multiple identical rounds (same q, sigma) to the accountant.
        q: client sampling rate in [0,1]
        sigma: noise multiplier (std dev = sigma * clipping C); give sigma as 'multiplier'
        steps: how many rounds to add at once
        """
        q = max(0.0, min(1.0, float(q)))
        sigma = float(sigma)
        steps = int(steps)

        if sigma <= 0 or steps <= 0 or q == 0.0:
            return  # nothing to add

        if self._rdp_available:
            # Use library subsampled Gaussian RDP
            if self._use_tf_privacy:
                compute_rdp = self._tfp_compute_rdp
            else:
                compute_rdp = self._opacus_compute_rdp

            rdp_inc = compute_rdp(
                q=q,
                noise_multiplier=sigma,
                steps=steps,
                orders=self.orders,
            )
            # rdp_inc is a list matching orders
            self.rdp_cumulative = [a + b for a, b in zip(self.rdp_cumulative, rdp_inc)]
        else:
            # Fallback (no subsampling): RDP of Gaussian mech (per step) at order alpha is alpha/(2sigma^2)
            # Then multiply by steps. This is a conservative upper bound when q<1.
            for i, alpha in enumerate(self.orders):
                if alpha <= 1:
                    continue
                self.rdp_cumulative[i] += steps * (alpha / (2.0 * sigma * sigma))

    def get_epsilon(self, delta: float) -> float:
        """
        Convert accumulated RDP to (epsilon, delta) and return epsilon.
        Uses library conversion if available, else a standard infimum over orders.
        """
        delta = float(delta)
        if delta <= 0 or sum(self.rdp_cumulative) == 0.0:
            return 0.0

        if self._rdp_available:
            if self._use_tf_privacy:
                get_priv = self._tfp_get_privacy_spent
                eps, _, _ = get_priv(orders=self.orders, rdp=self.rdp_cumulative, delta=delta)
                return float(eps)
            else:
                # Opacus get_privacy_spent returns (eps, alpha_opt)
                get_priv = self._opacus_get_privacy_spent
                result = get_priv(orders=self.orders, rdp=self.rdp_cumulative, delta=delta)
                if isinstance(result, tuple) and len(result) >= 2:
                    eps = result[0]
                    return float(eps)
                else:
                    eps = result  # single value
                    return float(eps)

        # Manual conversion: epsilon(alpha) = rdp(alpha) + log(1/delta)/(alpha-1), pick best alpha
        import math
        best = float("inf")
        for alpha, rdp_val in zip(self.orders, self.rdp_cumulative):
            if alpha <= 1:
                continue
            eps_a = rdp_val + math.log(1.0 / delta) / (alpha - 1.0)
            best = min(best, eps_a)
        return float(best)

    def get_status(self) -> dict:
        """
        Return current accounting status for debugging/logging
        """
        return {
            "rdp_available": self._rdp_available,
            "use_tf_privacy": self._use_tf_privacy,
            "use_opacus": self._use_opacus,
            "orders": self.orders,
            "rdp_cumulative": self.rdp_cumulative,
            "total_rdp": sum(self.rdp_cumulative)
        }

    def reset(self):
        """
        Reset the accountant to start fresh accounting
        """
        self.rdp_cumulative = [0.0] * len(self.orders)
