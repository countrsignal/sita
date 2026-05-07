import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_lr_schedule(
    base_lr=0.0,
    max_lr=1.8e-3,
    min_lr=1e-6,
    warmup_no_steps=1000,
    start_decay_after_n_steps=50000,
    decay_every_n_steps=50000,
    decay_factor=0.95,
    total_steps=200000,
    steps_per_epoch=1000,
    figsize=(12, 6),
    save_path=None
):
    # Create a dummy optimizer
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.1)  # initial lr doesn't matter
    
    # Create the scheduler
    scheduler = TransformerLRScheduler(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_no_steps=warmup_no_steps,
        start_decay_after_n_steps=start_decay_after_n_steps,
        decay_every_n_steps=decay_every_n_steps,
        decay_factor=decay_factor
    )
    
    # Track learning rates
    learning_rates = []
    
    # Simulate stepping the optimizer and scheduler
    for i in range(total_steps):
        optimizer.step()  # Dummy step, doesn't matter for this simulation
        scheduler.step()
        lr = scheduler.get_last_lr()[0]  # Get the learning rate
        learning_rates.append(lr)

    print("Max lr: ", max(learning_rates))
    print("Min lr: ", min(learning_rates[warmup_no_steps:]))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert steps to epochs for x-axis
    epochs = np.arange(total_steps) / steps_per_epoch
    
    # Plot learning rate curve with epochs on x-axis
    ax.plot(epochs, learning_rates, linewidth=2)
    
    # Add vertical lines for phase transitions (now in epoch units)
    warmup_epoch = warmup_no_steps / steps_per_epoch
    decay_start_epoch = start_decay_after_n_steps / steps_per_epoch
    
    ax.axvline(x=warmup_epoch, color='r', linestyle='--', alpha=0.7, 
               label=f'End of warmup (Epoch {warmup_epoch:.1f})')
    ax.axvline(x=decay_start_epoch, color='g', linestyle='--', alpha=0.7,
               label=f'Start of decay (Epoch {decay_start_epoch:.1f})')
    
    # Add decay steps markers if they're visible in the plot range
    if total_steps > start_decay_after_n_steps:
        decay_epochs = []
        for i, decay_step in enumerate(range(
            start_decay_after_n_steps + decay_every_n_steps, 
            total_steps, 
            decay_every_n_steps
        )):
            decay_epoch = decay_step / steps_per_epoch
            if i == 0:
                ax.axvline(x=decay_epoch, color='purple', linestyle=':', alpha=0.5,
                          label=f'Decay step (every {decay_every_n_steps/steps_per_epoch:.1f} epochs)')
            else:
                ax.axvline(x=decay_epoch, color='purple', linestyle=':', alpha=0.5)
    
    # Formatting
    ax.set_title('Transformer Learning Rate Schedule', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    
    # Set y-axis to use scientific notation in the form 1e-6, 1e-7, etc.
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    
    # Annotate phases
    mid_warmup_epoch = warmup_epoch / 2
    mid_plateau_epoch = (warmup_epoch + decay_start_epoch) / 2
    mid_decay_epoch = (decay_start_epoch + (total_steps / steps_per_epoch)) / 2
    
    ax.annotate('Warmup\n(Linear)', xy=(mid_warmup_epoch, max_lr/2), 
                xytext=(mid_warmup_epoch, max_lr/2), ha='center', fontsize=12)
    
    ax.annotate('Plateau\n(Constant)', xy=(mid_plateau_epoch, max_lr * 0.9), 
                xytext=(mid_plateau_epoch, max_lr * 0.9), ha='center', fontsize=12)
    
    ax.annotate('Decay\n(Exponential)', xy=(mid_decay_epoch, max_lr * 0.5), 
                xytext=(mid_decay_epoch, max_lr * 0.5), ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# Source: https://github.com/jwohlwend/boltz
class TransformerLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Implements the learning rate schedule defined AF3.

    A linear warmup is followed by a plateau at the maximum
    learning rate and then exponential decay. Note that the
    initial learning rate of the optimizer in question is
    ignored; use this class' base_lr parameter to specify
    the starting point of the warmup.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        base_lr: float = 0.0,
        max_lr: float = 1.8e-3,
        min_lr: float = 1e-6,
        warmup_no_steps: int = 1000,
        start_decay_after_n_steps: int = 50000,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        """Initialize the learning rate scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer.
        last_epoch : int, optional
            The last epoch, by default -1
        verbose : bool, optional
            Whether to print verbose output, by default False
        base_lr : float, optional
            The base learning rate, by default 0.0
        max_lr : float, optional
            The maximum learning rate, by default 1.8e-3
        warmup_no_steps : int, optional
            The number of warmup steps, by default 1000
        start_decay_after_n_steps : int, optional
            The number of steps after which to start decay, by default 50000
        decay_every_n_steps : int, optional
            The number of steps after which to decay, by default 50000
        decay_factor : float, optional
            The decay factor, by default 0.95

        """
        step_counts = {
            "warmup_no_steps": warmup_no_steps,
            "start_decay_after_n_steps": start_decay_after_n_steps,
        }

        for k, v in step_counts.items():
            if v < 0:
                msg = f"{k} must be nonnegative"
                raise ValueError(msg)

        if warmup_no_steps > start_decay_after_n_steps:
            msg = "warmup_no_steps must not exceed start_decay_after_n_steps"
            raise ValueError(msg)

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_no_steps = warmup_no_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.decay_every_n_steps = decay_every_n_steps
        self.decay_factor = decay_factor

        super().__init__(optimizer, last_epoch=last_epoch)

    def state_dict(self) -> dict:
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            msg = (
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )
            raise RuntimeError(msg)

        step_no = self.last_epoch

        if step_no <= self.warmup_no_steps:
            lr = self.base_lr + (step_no / self.warmup_no_steps) * self.max_lr
        elif step_no > self.start_decay_after_n_steps:
            steps_since_decay = step_no - self.start_decay_after_n_steps
            exp = (steps_since_decay // self.decay_every_n_steps) + 1
            lr = self.max_lr * (self.decay_factor**exp)
            lr = max(lr, self.min_lr)
        else:  # plateau
            lr = self.max_lr

        return [lr for group in self.optimizer.param_groups]