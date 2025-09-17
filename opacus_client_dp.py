"""
OpacusClientDP: A plug-and-play differential privacy tool for PyTorch models
This class wraps Opacus PrivacyEngine to provide easy-to-use sample-level DP

Supported Opacus versions: 1.4.0 - 1.5.0
"""
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants import RDPAccountant
import logging
from typing import Tuple, Optional, Dict, Any, Union, List
import warnings
from packaging import version
import opacus
from contextlib import contextmanager
from threading import Lock
import math


# --- Laplace Noise Mechanism for Opacus ---
class LaplaceNoiseMechanism:
    """
    Correct Laplace noise mechanism based on differential privacy theory
    """
    def __init__(self, epsilon_per_step: float, max_grad_norm: float, device=None):
        """
        Args:
            epsilon_per_step: Privacy budget epsilon per step
            max_grad_norm: L2 sensitivity (maximum gradient norm)
            device: Device (CPU/GPU)
        """
        self.epsilon_per_step = epsilon_per_step
        self.max_grad_norm = max_grad_norm
        self.device = device
        # Laplace distribution scale parameter b = sensitivity / epsilon
        self.scale = max_grad_norm / epsilon_per_step

    def add_noise(self, grad_sample: torch.Tensor) -> torch.Tensor:
        """
        Add Laplace noise to gradient samples
        """
        if self.epsilon_per_step == 0:
            return grad_sample
        
        # Generate Laplace noise
        laplace_dist = torch.distributions.Laplace(0, self.scale)
        noise = laplace_dist.sample(grad_sample.shape).to(grad_sample.device)
        
        return grad_sample + noise


class LaplacePrivacyAccountant:
    """
    Privacy accountant for Laplace mechanism
    """
    def __init__(self, epsilon_per_step: float, delta: float = 0.0):
        """
        Args:
            epsilon_per_step: Privacy budget per step
            delta: For pure epsilon-DP (Laplace mechanism), delta should be 0
        """
        self.epsilon_per_step = epsilon_per_step
        self.delta = delta
        self.steps_taken = 0
    
    def step(self):
        """Record one privacy consumption step"""
        self.steps_taken += 1
    
    def get_epsilon(self) -> float:
        """
        Get total privacy consumption
        For Laplace mechanism, epsilon accumulates linearly
        """
        return self.epsilon_per_step * self.steps_taken
    
    def reset(self):
        """Reset the accountant"""
        self.steps_taken = 0


# Version check
SUPPORTED_OPACUS_MIN = "1.4.0"
SUPPORTED_OPACUS_MAX = "1.5.999"

_opacus_version = version.parse(opacus.__version__)
if not (version.parse(SUPPORTED_OPACUS_MIN) <= _opacus_version <= version.parse(SUPPORTED_OPACUS_MAX)):
    warnings.warn(
        f"Opacus version {opacus.__version__} is not officially supported. "
        f"Supported versions: {SUPPORTED_OPACUS_MIN} - {SUPPORTED_OPACUS_MAX}. "
        f"Some features may not work correctly.",
        UserWarning
    )


class DPAttachmentError(Exception):
    """Raised when DP attachment fails"""
    pass


class DPConfigurationError(Exception):
    """Raised when DP configuration is invalid"""
    pass


class DPPrivacyAccountingError(Exception):
    """Raised when privacy accounting fails"""
    pass


class DPDistributedTrainingError(Exception):
    """Raised when distributed training is detected but not supported"""
    pass


class OpacusClientDP:
    """
    A wrapper class for Opacus PrivacyEngine that provides differential privacy
    for PyTorch models with minimal code changes.
    
    Note:
        - Only supports single-GPU training. For distributed training, use separate instances per process.
        - When using target_epsilon, the actual noise_multiplier cannot be reliably extracted.
        - Thread-safe for basic operations and concurrent access.
    
    Supported Opacus versions: 1.4.0 - 1.5.x
    """
    
    # Class-level lock for thread safety
    _lock = Lock()
    
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        epochs: int = 1,
        sample_rate: Optional[float] = None,
        secure_mode: bool = False,
        strict_mode: bool = True,
        experimental_mode: bool = False,
        log_level: str = "INFO",
        noise_mechanism: str = "gaussian",  # Supports 'gaussian' or 'laplace'
        epsilon_per_step: Optional[float] = None  # Privacy budget per step for Laplace mechanism
    ):
        """
        Initialize the differential privacy configuration
        
        Args:
            noise_multiplier: Noise multiplier for DP-SGD (only for Gaussian mechanism)
            max_grad_norm: Maximum gradient norm for clipping
            delta: Privacy parameter delta
            epochs: Number of training epochs
            sample_rate: Sampling rate (batch_size / dataset_size)
            secure_mode: Whether to use Opacus secure mode
            strict_mode: If True, raise exceptions instead of using defaults
            experimental_mode: Enable experimental features
            log_level: Logging level
            noise_mechanism: Noise mechanism type ('gaussian' or 'laplace')
            epsilon_per_step: Privacy budget per step for Laplace mechanism (only for Laplace mechanism)
        """
        # Validate parameters
        if noise_multiplier <= 0:
            raise DPConfigurationError("noise_multiplier must be positive")
        if max_grad_norm <= 0:
            raise DPConfigurationError("max_grad_norm must be positive")
        if delta <= 0 or delta >= 1:
            raise DPConfigurationError("delta must be in (0, 1)")
        if epochs <= 0:
            raise DPConfigurationError("epochs must be positive")
        if noise_mechanism.lower() not in ["gaussian", "laplace"]:
            raise DPConfigurationError("noise_mechanism must be 'gaussian' or 'laplace'")
        
        # Laplace mechanism requires epsilon_per_step
        if noise_mechanism.lower() == "laplace":
            if epsilon_per_step is None or epsilon_per_step <= 0:
                raise DPConfigurationError("epsilon_per_step must be provided and positive for Laplace mechanism")
        
        # Store initial configuration
        self._initial_config = {
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate,
            "max_grad_norm": max_grad_norm,
            "delta": delta,
            "epochs": epochs,
            "noise_mechanism": noise_mechanism.lower(),
            "epsilon_per_step": epsilon_per_step
        }
        
        # Configuration parameters
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.epochs = epochs
        self.sample_rate = sample_rate
        self.secure_mode = secure_mode
        self.strict_mode = strict_mode
        self.experimental_mode = experimental_mode
        self.noise_mechanism = noise_mechanism.lower()
        self.epsilon_per_step = epsilon_per_step
        
        # Internal state
        self.privacy_engine = None
        self.private_optimizer = None
        self.is_attached = False
        self.total_steps = 0
        self._target_epsilon = None
        self._actual_noise_multiplier = None
        self._distributed_warning_shown = False
        self._laplace_accountant = None  # Privacy accountant for Laplace mechanism
        
        # Instance-level lock for thread safety
        self._instance_lock = Lock()
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def __enter__(self):
        """Enter context manager - for future extensibility"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - automatically detach and cleanup"""
        if self.is_attached:
            self.detach()
        return False  # Don't suppress exceptions
    
    @contextmanager
    def training_session(self, model=None, optimizer=None, data_loader=None, **attach_kwargs):
        """
        Context manager for a complete training session with automatic attach/detach
        
        Usage:
            with dp_engine.training_session(model, optimizer, data_loader) as (priv_model, priv_opt, priv_dl):
                # Training code here
                pass
        """
        if model is None or optimizer is None or data_loader is None:
            # If no arguments provided, just enter/exit for cleanup
            try:
                yield self
            finally:
                if self.is_attached:
                    self.detach()
        else:
            # Full training session with attach/detach
            try:
                private_components = self.attach(model, optimizer, data_loader, **attach_kwargs)
                yield private_components
            finally:
                if self.is_attached:
                    self.detach()
    
    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        target_epsilon: Optional[float] = None,
        poisson_sampling: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
        """
        Attach differential privacy to model, optimizer, and data loader
        
        Args:
            model: PyTorch model to be made private
            optimizer: Optimizer for the model
            data_loader: Training data loader
            target_epsilon: Target epsilon value
            poisson_sampling: Whether to use Poisson sampling
        
        Returns:
            Tuple of (private_model, private_optimizer, private_data_loader)
        """
        with self._instance_lock:
            if self.is_attached:
                raise DPAttachmentError("Privacy engine is already attached. Call detach() first.")
            
            try:
                # Check for distributed training
                self._check_distributed_training(model, optimizer)
                
                # Validate inputs
                self._validate_inputs(model, optimizer, data_loader)
                
                # Fix model for Opacus compatibility; keep device consistent
                model, was_fixed = self._validate_and_fix_model(model)
                
                # Calculate or validate sample rate
                if self.sample_rate is None:
                    calculated_rate = self._calculate_sample_rate(data_loader)
                    if self.strict_mode:
                        raise DPConfigurationError(
                            f"sample_rate must be explicitly provided in strict mode. "
                            f"Calculated value would be: {calculated_rate:.6f}"
                        )
                    else:
                        self.sample_rate = calculated_rate
                        warnings.warn(
                            f"sample_rate not provided, using calculated value: {calculated_rate:.6f}. "
                            f"This may affect privacy guarantees. Consider providing explicit sample_rate.",
                            UserWarning
                        )
                
                # If model was fixed, rebuild optimizer to bind new parameters
                if 'was_fixed' in locals() and was_fixed:
                    try:
                        opt_cls = type(optimizer)
                        opt_kwargs = {}
                        # Try to reuse optimizer defaults like lr, momentum, weight_decay
                        if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                            pg = optimizer.param_groups[0]
                            for k in ['lr', 'momentum', 'weight_decay', 'betas', 'eps']:
                                if k in pg:
                                    opt_kwargs[k] = pg[k]
                        optimizer = opt_cls(model.parameters(), **opt_kwargs)
                        self.logger.info("Optimizer reinitialized to bind parameters of fixed model.")
                    except Exception as e:
                        raise DPAttachmentError(f"Failed to reinitialize optimizer after model fix: {str(e)}") from e

                # Create PrivacyEngine
                self.privacy_engine = PrivacyEngine(
                    secure_mode=self.secure_mode
                    )
                
                # Store target epsilon if provided
                self._target_epsilon = target_epsilon
                
                # Make model, optimizer, and data_loader private
                if target_epsilon is not None:
                    # Use target epsilon to auto-calculate noise multiplier
                    private_model, private_optimizer, private_data_loader = \
                        self.privacy_engine.make_private_with_epsilon(
                            module=model,
                            optimizer=optimizer,
                            data_loader=data_loader,
                            epochs=self.epochs,
                            target_epsilon=target_epsilon,
                            target_delta=self.delta,
                            max_grad_norm=self.max_grad_norm,
                            poisson_sampling=poisson_sampling
                        )
                    
                    # Try to extract noise multiplier in experimental mode
                    if self.experimental_mode:
                        self._actual_noise_multiplier = self._extract_noise_multiplier_experimental()
                        if self._actual_noise_multiplier:
                            self.logger.info(
                                f"Target epsilon: {target_epsilon:.4f}, "
                                f"Extracted noise multiplier: {self._actual_noise_multiplier:.4f} (experimental)"
                            )
                    else:
                        self.logger.warning(
                            f"Using target_epsilon={target_epsilon:.4f}. "
                            f"Actual noise_multiplier cannot be reliably determined. "
                            f"Enable experimental_mode to attempt extraction."
                        )
                else:
                    # Use provided noise multiplier
                    private_model, private_optimizer, private_data_loader = \
                        self.privacy_engine.make_private(
                            module=model,
                            optimizer=optimizer,
                            data_loader=data_loader,
                            noise_multiplier=self.noise_multiplier,
                            max_grad_norm=self.max_grad_norm,
                            poisson_sampling=poisson_sampling
                        )
                    
                    self._actual_noise_multiplier = self.noise_multiplier
                
                # If using Laplace noise, need to replace optimizer's noise mechanism
                if self.noise_mechanism == "laplace":
                    self._apply_laplace_noise_mechanism(private_optimizer)
                    self.logger.info(f"Applied Laplace noise mechanism with scale: {self._actual_noise_multiplier * self.max_grad_norm:.4f}")
                
                # Store reference to private optimizer
                self.private_optimizer = private_optimizer
                self.is_attached = True
                
                # Log attachment info
                noise_info = (f"noise_multiplier: {self._actual_noise_multiplier:.4f}" 
                             if self._actual_noise_multiplier 
                             else "noise_multiplier: unknown (using target_epsilon)")
                
                self.logger.info(
                    f"Differential privacy successfully attached - "
                    f"{noise_info}, "
                    f"max_grad_norm: {self.max_grad_norm:.3f}, "
                    f"sample_rate: {self.sample_rate:.6f}, "
                    f"noise_mechanism: {self.noise_mechanism}, "
                    f"secure_mode: {self.secure_mode}"
                )
                
                return private_model, private_optimizer, private_data_loader
                
            except Exception as e:
                self.logger.error(f"Failed to attach differential privacy: {str(e)}")
                if self.strict_mode:
                    raise DPAttachmentError(f"Cannot attach differential privacy: {str(e)}") from e
                else:
                    raise
    
    def detach(self):
        """
        Detach and cleanup the privacy engine
        """
        with self._instance_lock:
            if not self.is_attached:
                self.logger.warning("Privacy engine is not attached, nothing to detach")
                return
            
            try:
                # Clean up references
                self.privacy_engine = None
                self.private_optimizer = None
                self.is_attached = False
                
                self.logger.info("Differential privacy successfully detached and cleaned up")
                
            except Exception as e:
                self.logger.error(f"Error during detach: {str(e)}")
                # Force cleanup even if error occurs
                self.privacy_engine = None
                self.private_optimizer = None
                self.is_attached = False
    
    def _validate_inputs(self, model: nn.Module, optimizer: torch.optim.Optimizer, data_loader: torch.utils.data.DataLoader):
        """Internal helper to validate input types and values"""
        if not isinstance(model, nn.Module):
            raise DPConfigurationError("Model must be a PyTorch nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise DPConfigurationError("Optimizer must be a PyTorch optimizer")
        if not isinstance(data_loader, torch.utils.data.DataLoader):
            raise DPConfigurationError("DataLoader must be a PyTorch DataLoader")
        
        # Check if sample_rate is valid (if not strict_mode, it might be None)
        if self.sample_rate is not None and not (0 < self.sample_rate <= 1):
            raise DPConfigurationError("sample_rate must be between 0 and 1 (exclusive of 0)")
        
        # Check if epochs are positive
        if self.epochs <= 0:
            raise DPConfigurationError("epochs must be positive")

    def _validate_and_fix_model(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """
        Validate the model for Opacus compatibility and fix if necessary.

        Returns:
            (possibly_fixed_model, was_fixed)
        """
        # Remember original device to preserve placement after potential fix
        try:
            first_param = next(model.parameters())
            orig_device = first_param.device
        except StopIteration:
            orig_device = torch.device("cpu")

        if not ModuleValidator.is_valid(model):
            self.logger.warning("Model is not DP-compatible. Attempting to fix...")
            try:
                fixed_model = ModuleValidator.fix(model)
                # Ensure the fixed model stays on the original device
                fixed_model = fixed_model.to(orig_device)
                self.logger.info("Model successfully fixed for DP compatibility and moved to original device.")
                return fixed_model, True
            except Exception as e:
                raise DPAttachmentError(f"Failed to fix model for DP compatibility: {str(e)}") from e
        return model, False
    
    def _calculate_sample_rate(self, data_loader: torch.utils.data.DataLoader) -> float:
        """Calculate sample rate from data loader"""
        dataset_size = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        return batch_size / dataset_size
    
    def _apply_laplace_noise_mechanism(self, private_optimizer):
        """
        Apply Laplace noise mechanism to private optimizer
        Key correction: Add noise before parameter updates
        """
        # Create Laplace noise mechanism instance
        laplace_mechanism = LaplaceNoiseMechanism(
            epsilon_per_step=self.epsilon_per_step,
            max_grad_norm=self.max_grad_norm
        )
        
        # Create Laplace privacy accountant
        self._laplace_accountant = LaplacePrivacyAccountant(
            epsilon_per_step=self.epsilon_per_step,
            delta=0.0  # Laplace mechanism is pure epsilon-DP
        )
        
        # Save original step method
        original_step = private_optimizer.step
        
        def laplace_step(closure=None):
            """
            Correct Laplace noise optimizer step method
            """
            # First disable Opacus Gaussian noise
            if hasattr(private_optimizer, 'noise_multiplier'):
                private_optimizer.noise_multiplier = 0
            
            # Add Laplace noise to gradients before parameter updates
            for param_group in private_optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        # Apply gradient clipping (if needed)
                        grad_norm = torch.norm(param.grad)
                        if grad_norm > self.max_grad_norm:
                            param.grad = param.grad * (self.max_grad_norm / grad_norm)
                        
                        # Add Laplace noise
                        param.grad = laplace_mechanism.add_noise(param.grad)
            
            # Now call original step method with noisy gradients for parameter updates
            result = original_step(closure)
            
            # Record privacy consumption
            self._laplace_accountant.step()
            
            return result
        
        # Replace step method
        private_optimizer.step = laplace_step
    
    def _check_distributed_training(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Check if model or optimizer is configured for distributed training"""
        # Check for DataParallel
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            raise DPDistributedTrainingError(
                "Distributed training detected. OpacusClientDP currently only supports single-GPU training."
            )
        
        # Check for DDP in model modules
        for module in model.modules():
            if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                raise DPDistributedTrainingError(
                    f"Distributed module found: {type(module).__name__}. "
                    f"Please use single-GPU model."
                )
        
        # Warn about potential distributed setup
        if torch.cuda.device_count() > 1 and not self._distributed_warning_shown:
            warnings.warn(
                "Multiple GPUs detected. OpacusClientDP will only use the current device.",
                UserWarning
            )
            self._distributed_warning_shown = True
    
    def _extract_noise_multiplier_experimental(self) -> Optional[float]:
        """
        Experimental: Try to extract actual noise multiplier from Opacus internals
        """
        try:
            # Check accountant history
            if hasattr(self.privacy_engine, 'accountant'):
                accountant = self.privacy_engine.accountant
                
                # Try to get from RDPAccountant
                if isinstance(accountant, RDPAccountant):
                    if hasattr(accountant, '_noise_multiplier'):
                        return float(accountant._noise_multiplier)
            
            # Check optimizer attributes
            if self.private_optimizer:
                for attr in ['noise_multiplier', '_noise_multiplier']:
                    if hasattr(self.private_optimizer, attr):
                        value = getattr(self.private_optimizer, attr)
                        if value is not None:
                            return float(value)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Experimental noise multiplier extraction failed: {str(e)}")
            return None
    
    def step(self) -> int:
        """
        Increment step counter - MUST be called after each optimizer.step()
        
        Returns:
            Current total steps
        """
        if not self.is_attached:
            raise RuntimeError("Cannot call step() before attach() or after detach()")
        
        with self._instance_lock:
            self.total_steps += 1
            
            # For Laplace mechanism, privacy accounting is handled in optimizer.step()
            # Here we only need to update total steps
        
        # Log periodically
        if self.total_steps % 100 == 0:
            self.logger.debug(f"Training step {self.total_steps} completed")
        
        return self.total_steps
    
    def get_privacy_spent(self, steps: Optional[int] = None) -> Dict[str, Any]:
        if not self.is_attached or self.privacy_engine is None:
            if self.strict_mode:
                raise DPPrivacyAccountingError("Privacy engine not attached")
            else:
                return {"epsilon": None, "delta": self.delta, "steps": 0, "noise_multiplier": None,
                        "error": "Privacy engine not attached"}

        try:
            if self.noise_mechanism == "laplace":
                # Use correct privacy accounting for Laplace mechanism
                if self._laplace_accountant is None:
                    raise DPPrivacyAccountingError("Laplace accountant not initialized")
                
                epsilon = self._laplace_accountant.get_epsilon()
                return {
                    "epsilon": float(epsilon),
                    "delta": 0.0,  # Laplace mechanism is pure epsilon-DP
                    "epsilon_per_step": self.epsilon_per_step,
                    "max_grad_norm": self.max_grad_norm,
                    "sample_rate": self.sample_rate,
                    "steps": self.total_steps,
                    "noise_mechanism": "laplace",
                    "scale_parameter": self.max_grad_norm / self.epsilon_per_step,
                    "opacus_version": opacus.__version__,
                }
            else:
                # Use RDP accounting for Gaussian mechanism
                epsilon = self._get_epsilon_by_version()
                return {
                    "epsilon": float(epsilon),
                    "delta": self.delta,
                    "noise_multiplier": self._actual_noise_multiplier,
                    "max_grad_norm": self.max_grad_norm,
                    "sample_rate": self.sample_rate,
                    "steps": self.total_steps,
                    "target_epsilon": self._target_epsilon,
                    "noise_mechanism": "gaussian",
                    "opacus_version": opacus.__version__,
                }
        except Exception as e:
            error_msg = f"Privacy accounting failed: {str(e)}"
            self.logger.error(error_msg)
            
            if self.strict_mode:
                raise DPPrivacyAccountingError(error_msg) from e
            else:
                return {
                    "epsilon": None,
                    "delta": self.delta,
                    "steps": self.total_steps,
                    "noise_multiplier": self._actual_noise_multiplier,
                    "error": error_msg
                }
    
    def _get_epsilon_by_version(self) -> float:
        if _opacus_version >= version.parse("1.4.1"):
            try:
                return self.privacy_engine.get_epsilon(delta=self.delta)
            except (AttributeError, TypeError):
                return self.privacy_engine.accountant.get_epsilon(delta=self.delta)
        else:
            return self.privacy_engine.accountant.get_epsilon(delta=self.delta)
    
    def reset(self, keep_config: bool = False):
        """
        Reset the privacy engine state
        """
        with self._instance_lock:
            # Detach if attached
            if self.is_attached:
                self.detach()
            
            # Reset state
            self.total_steps = 0
            self._target_epsilon = None
            self._actual_noise_multiplier = None
            
            # Reset Laplace accountant
            if self._laplace_accountant:
                self._laplace_accountant.reset()
            
            if not keep_config:
                # Reset to initial configuration
                self.noise_multiplier = self._initial_config["noise_multiplier"]
                self.sample_rate = self._initial_config["sample_rate"]
                self.max_grad_norm = self._initial_config["max_grad_norm"]
                self.delta = self._initial_config["delta"]
                self.epochs = self._initial_config["epochs"]
                self.noise_mechanism = self._initial_config["noise_mechanism"]
                self.epsilon_per_step = self._initial_config["epsilon_per_step"]
            
            self.logger.debug(f"Privacy engine state reset (keep_config={keep_config})")
    
    @contextmanager
    def auto_step(self, optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]):
        """
        Context manager for automatic step counting
        """
        # Handle single optimizer
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        
        # Store original step methods
        original_steps = []
        for opt in optimizers:
            original_steps.append(opt.step)
        
        # Replace with wrapped versions
        def make_wrapped_step(original_step_fn):
            def wrapped_step(*args, **kwargs):
                result = original_step_fn(*args, **kwargs)
                self.step()
                return result
            return wrapped_step
        
        try:
            for i, opt in enumerate(optimizers):
                opt.step = make_wrapped_step(original_steps[i])
            
            yield
            
        finally:
            # Restore original step methods
            for opt, original_step in zip(optimizers, original_steps):
                opt.step = original_step
    
    def estimate_epsilon(
        self,
        steps: int,
        noise_multiplier: Optional[float] = None,
        sample_rate: Optional[float] = None,
        delta: Optional[float] = None
    ) -> float:
        """
        Estimate epsilon for given parameters without training
        """
        noise_multiplier = noise_multiplier or self.noise_multiplier
        sample_rate = sample_rate or self.sample_rate
        delta = delta or self.delta
        
        if sample_rate is None:
            raise ValueError("sample_rate must be provided or set in the instance")
        
        # Create temporary accountant for estimation
        from opacus.accountants import create_accountant
        
        accountant = create_accountant(mechanism="rdp")
        
        # Simulate steps
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        
        return accountant.get_epsilon(delta=delta)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current DP configuration"""
        config = {
            "noise_multiplier": self.noise_multiplier,
            "actual_noise_multiplier": self._actual_noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "delta": self.delta,
            "epochs": self.epochs,
            "sample_rate": self.sample_rate,
            "secure_mode": self.secure_mode,
            "strict_mode": self.strict_mode,
            "experimental_mode": self.experimental_mode,
            "is_attached": self.is_attached,
            "total_steps": self.total_steps,
            "opacus_version": opacus.__version__,
            "supported_versions": f"{SUPPORTED_OPACUS_MIN} - {SUPPORTED_OPACUS_MAX}",
            "noise_mechanism": self.noise_mechanism
        }
        
        if self.noise_mechanism == "laplace":
            config["epsilon_per_step"] = self.epsilon_per_step
            config["laplace_scale"] = self.max_grad_norm / self.epsilon_per_step if self.epsilon_per_step else None
        
        if self._target_epsilon is not None:
            config["target_epsilon"] = self._target_epsilon
        
        return config
    
    def __repr__(self) -> str:
        nm = f"{self._actual_noise_multiplier:.4f}" if self._actual_noise_multiplier is not None else "Unknown"
        sr = f"{self.sample_rate:.6f}" if self.sample_rate is not None else "None"
        return (
            "OpacusClientDP(\n"
            f"  noise_multiplier={self.noise_multiplier:.4f},\n"
            f"  actual_noise_multiplier={nm},\n"
            f"  max_grad_norm={self.max_grad_norm:.3f},\n"
            f"  delta={self.delta:.2e},\n"
            f"  sample_rate={sr},\n"
            f"  total_steps={self.total_steps},\n"
            f"  is_attached={self.is_attached},\n"
            f"  strict_mode={self.strict_mode},\n"
            f"  noise_mechanism={self.noise_mechanism},\n"
            f"  opacus_version={opacus.__version__}\n"
            ")"
        )
    
    def __str__(self) -> str:
        """String representation"""
        noise_str = (f"{self._actual_noise_multiplier:.4f}" 
                    if self._actual_noise_multiplier 
                    else f"{self.noise_multiplier:.4f}*")
        
        return (
            f"OpacusClientDP("
            f"σ={noise_str}, "
            f"C={self.max_grad_norm}, "
            f"δ={self.delta:.1e}, "
            f"steps={self.total_steps}, "
            f"noise={self.noise_mechanism}, "
            f"attached={self.is_attached})"
        )