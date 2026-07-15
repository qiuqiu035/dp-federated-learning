"""
Adapter: Make the new LocalDpFixedMod compatible with the existing serverless architecture,
and integrate automatic noise computation functionality.
"""
import math
import numpy as np
import torch
from flwr.common import FitIns, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.client import Client

from localdpmod.differential_privacy import (
    add_localdp_fixed_gaussian_noise_to_params,
    # add_localdp_l1laplace_noise_to_params,
    compute_clip_model_update,
    # compute_clip_model_update_l1,
    add_localdp_l2laplace_noise_to_params,
)

# Import seed manager for reproducible noise generation
from seed_manager import get_seed_manager



class LocalDpModAdapter:
    """
    Adapter class: Make the new LocalDpFixedMod functionality compatible with the existing serverless architecture.

    This class simulates the core functionality of LocalDpFixedMod but uses a wrap mode.
    """

    def __init__(
        self,
        clipping_norm: float,
        mechanism: str = "gaussian",
        sensitivity: float = 1.0,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_list=None,
        partition_id: int = 0,
        batch_size: int = 64,
        local_steps: int = None
    ):
        self.clipping_norm = clipping_norm
        self.mechanism = mechanism
        self.sensitivity = 1.0
        self.epsilon = epsilon
        self.delta = delta
        self.noise_list = noise_list or [0.1]
        self.partition_id = partition_id
        self.batch_size = batch_size
        self.local_steps = local_steps

        if mechanism not in ["gaussian", "laplace"]:
            raise ValueError("Mechanism must be either 'gaussian' or 'laplace'.")

    def _scaled_std(self, base_std: float) -> float:
        """
        Compute the noise standard deviation - fixed for client-level DP (no scaling)

        Note: Removed scale_mode related logic to ensure client-level DP's sensitivity is consistent with noise ratio
        """
        return base_std

    def wrap(self, client: Client) -> Client:
        """Wrap the client to add DP functionality"""

        class _DpClient(Client):
            def __init__(self, base: Client, mod: "LocalDpModAdapter"):
                self.base = base
                self.mod = mod

            def get_properties(self, ins):
                return self.base.get_properties(ins)

            def get_parameters(self, ins):
                return self.base.get_parameters(ins)

            def fit(self, ins: FitIns) -> FitRes:
                server_to_client_params = parameters_to_ndarrays(ins.parameters)

                res: FitRes = self.base.fit(ins)

                if res.parameters is None:
                    return res

                client_to_server_params = parameters_to_ndarrays(res.parameters)

                compute_clip_model_update(
                    client_to_server_params,
                    server_to_client_params,
                    self.mod.clipping_norm,
                )

                res.parameters = ndarrays_to_parameters(client_to_server_params)

                cfg = ins.config or {}
                mech = (cfg.get("dp_mechanism", self.mod.mechanism) or "gaussian").lower()
                mode = (cfg.get("dp_mode", getattr(self.mod, "mode", "client")) or "client").lower()
                C = float(cfg.get("max_grad_norm", self.mod.clipping_norm or 1.0))
                eps_round = float(cfg.get("round_target_epsilon", self.mod.epsilon or 0.0) or 0.0)
                delta = float(cfg.get("delta") or 1e-9)

                # Derive reproducible noise from the client and round identifiers.
                client_id = cfg.get("partition_id", self.mod.partition_id)
                round_num = cfg.get("server_round", 0)

                # Set the DP noise seed through the shared seed manager.
                seed_mgr = get_seed_manager()
                noise_seed = seed_mgr.get_dp_noise_seed(client_id, round_num)
                torch.manual_seed(noise_seed)
                np.random.seed(noise_seed)
                print(f"[LocalDpModAdapter]  Client {client_id} Round {round_num}: Using DP noise seed {noise_seed} (MASTER_SEED={seed_mgr.get_master_seed()})")

                if mech == "gaussian":
                    if mode == "client":
                        nm = cfg.get("noise_multiplier", None)
                        if nm is None:
                            noise_stddev = C * np.sqrt(2.0 * np.log(1.25 / max(1e-12, delta))) / max(1e-12, eps_round)
                        else:
                            noise_stddev = C * float(nm)
                    else:
                        noise_stddev = C * np.sqrt(2.0 * np.log(1.25 / max(1e-12, delta))) / max(1e-12, eps_round)

                    param_arrays = parameters_to_ndarrays(res.parameters)
                    flat_params = np.concatenate([p.flatten() for p in param_arrays])
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                    flat_tensor = torch.tensor(flat_params, device=device)
                    noise = torch.randn_like(flat_tensor) * noise_stddev
                    flat_noisy = flat_tensor + noise

                    idx = 0
                    noisy_arrays = []
                    for p in param_arrays:
                        size = p.size
                        noisy_arrays.append(flat_noisy[idx:idx+size].cpu().numpy().reshape(p.shape))
                        idx += size

                    res.parameters = ndarrays_to_parameters(noisy_arrays)

                    # Verify the values that will be returned to the server.
                    final_noisy = parameters_to_ndarrays(res.parameters)
                    flat_final = np.concatenate([p.ravel() for p in final_noisy])
                    flat_clip = np.concatenate([p.ravel() for p in param_arrays])
                    final_noisy_t = torch.tensor(flat_final, device=device)
                    flat_clip_t = torch.tensor(flat_clip, device=device)

                    noise_only_l2 = float(torch.norm(final_noisy_t - flat_clip_t, p=2).cpu().item())
                    clip_l2 = float(torch.norm(flat_clip_t, p=2).cpu().item())
                    total_l2 = float(torch.norm(final_noisy_t, p=2).cpu().item())
                    K = int(cfg.get("num_clients_in_round", 100))

                    print(f"[LocalDpModAdapter][FINAL] Client {client_id} Round {round_num}")
                    print(f"  sigma={noise_stddev:.6f}, C={C}, eps_round={eps_round}, delta={delta}")
                    print(f"  ||noise_only||_2={noise_only_l2:.4e}, ||clipped||_2={clip_l2:.4e}, ||final||_2={total_l2:.4e}")
                    print(f"  Expect agg noise L2 approximately {noise_only_l2/np.sqrt(max(K,1)):.4e} (K={K})")

                    # Warn when the observed perturbation is unexpectedly small.
                    if noise_only_l2 < 2.0 * C:
                        print(f"[LocalDpModAdapter][FINAL][SUSPICIOUS] Noise too small vs C, DP may not be applied!")
                        print(f"  ||noise_only||_2={noise_only_l2:.4e} < 2xC={2.0*C:.4e}")
                        if res.metrics is None:
                            res.metrics = {}
                        res.metrics["dp_applied"] = False
                        res.metrics["suspicious_noise"] = True
                    else:
                        print(f"[LocalDpModAdapter][FINAL][] DP verified: ||noise_only||_2={noise_only_l2:.4e} > 2xC={2.0*C:.4e}")
                        if res.metrics is None:
                            res.metrics = {}
                        res.metrics["dp_applied"] = True

                    # Expose diagnostics for server-side verification.
                    if res.metrics is None:
                        res.metrics = {}
                    res.metrics["final_noise_l2"] = float(noise_only_l2)
                    res.metrics["final_signal_l2"] = float(clip_l2)
                    res.metrics["final_total_l2"] = float(total_l2)
                    res.metrics["expect_noise_after_agg"] = float(noise_only_l2 / np.sqrt(max(K, 1)))

                elif mech == "laplace":
                    b = C / max(1e-12, eps_round)

                    param_arrays = parameters_to_ndarrays(res.parameters)
                    flat_params = np.concatenate([p.flatten() for p in param_arrays])
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                    flat_tensor = torch.tensor(flat_params, device=device)
                    laplace = torch.distributions.Laplace(
                        loc=torch.tensor(0.0, device=device),
                        scale=torch.tensor(b, device=device)
                    )
                    noise = laplace.sample(flat_tensor.shape)
                    flat_noisy = flat_tensor + noise

                    idx = 0
                    noisy_arrays = []
                    for p in param_arrays:
                        size = p.size
                        noisy_arrays.append(flat_noisy[idx:idx+size].cpu().numpy().reshape(p.shape))
                        idx += size

                    res.parameters = ndarrays_to_parameters(noisy_arrays)

                    # Verify the final Laplace-noised update.
                    final_noisy = parameters_to_ndarrays(res.parameters)
                    flat_final = np.concatenate([p.ravel() for p in final_noisy])
                    flat_clip = np.concatenate([p.ravel() for p in param_arrays])
                    final_noisy_t = torch.tensor(flat_final, device=device)
                    flat_clip_t = torch.tensor(flat_clip, device=device)

                    noise_only_l2 = float(torch.norm(final_noisy_t - flat_clip_t, p=2).cpu().item())
                    clip_l2 = float(torch.norm(flat_clip_t, p=2).cpu().item())
                    total_l2 = float(torch.norm(final_noisy_t, p=2).cpu().item())
                    K = int(cfg.get("num_clients_in_round", 100))

                    print(f"[LocalDpModAdapter][FINAL] Client {client_id} Round {round_num}")
                    print(f"  b={b:.6f}, C={C}, eps_round={eps_round}")
                    print(f"  ||noise_only||_2={noise_only_l2:.4e}, ||clipped||_2={clip_l2:.4e}, ||final||_2={total_l2:.4e}")
                    print(f"  Expect agg noise L2 approximately {noise_only_l2/np.sqrt(max(K,1)):.4e} (K={K})")

                    if noise_only_l2 < 2.0 * C:
                        print(f"[LocalDpModAdapter][FINAL][SUSPICIOUS] Noise too small vs C, DP may not be applied!")
                        print(f"  ||noise_only||_2={noise_only_l2:.4e} < 2xC={2.0*C:.4e}")
                        if res.metrics is None:
                            res.metrics = {}
                        res.metrics["dp_applied"] = False
                        res.metrics["suspicious_noise"] = True
                    else:
                        print(f"[LocalDpModAdapter][FINAL][] DP verified: ||noise_only||_2={noise_only_l2:.4e} > 2xC={2.0*C:.4e}")
                        if res.metrics is None:
                            res.metrics = {}
                        res.metrics["dp_applied"] = True

                    if res.metrics is None:
                        res.metrics = {}
                    res.metrics["final_noise_l2"] = float(noise_only_l2)
                    res.metrics["final_signal_l2"] = float(clip_l2)
                    res.metrics["final_total_l2"] = float(total_l2)
                    res.metrics["expect_noise_after_agg"] = float(noise_only_l2 / np.sqrt(max(K, 1)))

                if res.metrics is None:
                    res.metrics = {}

                res.metrics.update({
                    "noise_mechanism": mech,
                    "dp_mode": mode,
                    "round_target_epsilon": float(eps_round),
                    "max_grad_norm": C,
                })

                if mech == "gaussian":
                    res.metrics["delta"] = float(delta)
                    if mode == "client":
                        nm = cfg.get("noise_multiplier")
                        if nm is not None:
                            res.metrics["noise_multiplier"] = float(nm)
                        num_clients_in_round = cfg.get("num_clients_in_round", 1)
                        total_clients = cfg.get("total_clients", num_clients_in_round)
                        client_sample_rate = num_clients_in_round / total_clients
                        res.metrics["client_sample_rate"] = float(client_sample_rate)
                elif mech == "laplace":
                    res.metrics["consumed_epsilon"] = float(eps_round)

                return res

            def evaluate(self, ins):
                return self.base.evaluate(ins)

        return _DpClient(client, self)


def create_localdp_adapter(
    mechanism: str,
    max_grad_norm: float,
    epsilon: float,
    delta: float = 1e-5,
    partition_id: int = 0,
    noise_list=None,
    auto_calculate_noise: bool = True,
    noise_multiplier: float = None,
) -> LocalDpModAdapter:
    """
    Create a convenient function for building a LocalDpMod adapter (supports automatic noise computation).
    Args:
        mechanism: "gaussian" or "laplace"
        max_grad_norm: Gradient clipping threshold (C)
        epsilon: Privacy budget
        delta: delta parameter for the Gaussian mechanism
        partition_id: Client partition ID
        noise_list: List of noise values (for compatibility with the original interface)
        auto_calculate_noise: Whether to automatically compute noise parameters
        noise_multiplier: Noise multiplier for Gaussian mechanism (sigma = C * noise_multiplier)
                         If provided, this overrides auto_calculate_noise for Gaussian

        Returns:
        A LocalDpModAdapter instance
    """

    # Prefer an explicitly calibrated multiplier for the Gaussian mechanism.
    if mechanism == "gaussian" and noise_multiplier is not None:
        auto_noise = max_grad_norm * noise_multiplier
        print(f" Using explicit noise_multiplier: sigma={auto_noise:.6f} (C={max_grad_norm} x nm={noise_multiplier:.6f})")
        if noise_list is None:
            noise_list = [auto_noise] * 500

    elif auto_calculate_noise:
        if mechanism == "gaussian":
            auto_noise = (max_grad_norm *
                         math.sqrt(2 * math.log(1.25 / delta)) /
                         epsilon)
            print(f" Auto-calculated Gaussian noise: sigma={auto_noise:.4f} for epsilon={epsilon}")

        elif mechanism == "laplace":
            auto_noise = max_grad_norm / epsilon
            print(f" Auto-calculated Laplace noise: b={auto_noise:.4f} for epsilon={epsilon}")

        if noise_list is None:
            noise_list = [auto_noise] * 500

    return LocalDpModAdapter(
        clipping_norm=max_grad_norm,
        mechanism=mechanism,
        sensitivity=1.0,
        epsilon=epsilon,
        delta=delta,
        noise_list=noise_list,
        partition_id=partition_id,
    )
