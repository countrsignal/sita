import dgl
import torch
import numpy as np
import rootutils
import mdtraj as md
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from lightning import seed_everything

rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

from aita.data.molecule import ADP
from aita.pipeline.pipeline import Pipeline
from aita.plans import TrigPlan
from aita.interpolants import Interpolant
from aita.utils.interactive import view_mol_3d
from aita.utils.graph_utils import GraphAdapter
from aita.data.datasets import SimulationDataset
from aita.utils.rigid_align_loss import compute_mse_loss
from aita.models.atomic_transformer import AtomicTransformer

from aita.utils.interactive import (
    initialize_config,
    hyrda_init,
    view_mol_3d,
)


MOLECULE = "alanine_dipeptide"
FIG_DIR = "/net/galaxy/home/koes/dap181/labspace/aita/scripts/"
DATASET_DIR = "/net/galaxy/home/koes/dap181/labspace/aita/data/"


class InferenceWrapper(torch.nn.Module):

    def __init__(
        self,
        network: torch.nn.Module,
    ):
        super(InferenceWrapper, self).__init__()
        self.network = network
        self.graph_adapter = None


    def inference_fwd(self, g: dgl.DGLGraph):

        if self.graph_adapter is None:
            self.graph_adapter, *data = GraphAdapter.adapt_and_pad(g, target_key=None, apply_random_rotations=False, use_rbf=True)
        else:
            data = self.graph_adapter.graph_to_padded_tensor(g, target_key=None, apply_random_rotations=False, use_rbf=True)

        data = tuple(t.cuda() for t in data)
        times, xt, node_feats, atom_index, edge_feats, node_mask, edge_mask = data

        # NOTE: We do not need to reshape the velocity tensor because we expect that we only generate ONE molecular species at test-time!!
        velocity, x_h, edge_repr = self.network(
            x_t = xt,
            time = times,
            attr = node_feats,
            atom_index = atom_index,
            pair_feats = edge_feats,
            atom_mask = node_mask,
            pair_mask = edge_mask,
        )
        return velocity


def main():
    seed_everything(42)
    torch.set_float32_matmul_precision("high")

    ref_traj = md.load(f"{DATASET_DIR}/mds/temperature/{MOLECULE}_1200K_short_trajectory.dcd", top=f"{DATASET_DIR}/debug/{MOLECULE}.pdb")
    ref_coords = torch.from_numpy(ref_traj.xyz).reshape(-1, 66)

    interpolant = Interpolant(plan=TrigPlan(coupling_plan="ot"))
    ds = SimulationDataset(
        data_path=DATASET_DIR,
        param="1200K",
        anneal_type="temperature",
        split_json_filename=None,
        debug_molecule=MOLECULE,
    )
    dl = ds.get_train_dataloader(batch_size=512, num_workers=2, pin_memory=False)
    model = AtomicTransformer(
        node_feats_in=24,
        edge_feats_in=28,
        n_vecs=32,
        c_atoms=128,
        c_pairs=128,
        n_heads = 16,
        n_layers = 7,
        dropout_prob = 0.1,
        bias = False,
        initial_norm = True,
    )
    model = model.to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    n_epochs = 30
    device = "cuda:0"
    plot_interval = 5  # Plot loss every K epochs
    monitor = {"loss":  [], "lr": []}
    epoch_history = []
    for epoch in range(n_epochs):
        model.train()
        running_losses = {"loss":  0.0}
        num_batches = 0

        # Wrap dataloader with tqdm to show progress
        progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for g in progress_bar:
            
            # Generate noisy input: x_t = x1 + sigma(t) * z
            g = interpolant.plan(g)
            adapter, *data = GraphAdapter.adapt_and_pad(g, apply_random_rotations=True, use_rbf=True)
            data = tuple(t.cuda() for t in data)
            vt, times, xt, node_feats, atom_index, edge_feats, node_mask, edge_mask = data
            
            # Forward pass: predict x1 from (x_t, t)
            velocity, x_h, edge_repr = model(
                x_t = xt,
                time = times,
                attr = node_feats,
                atom_index = atom_index,
                pair_feats = edge_feats,
                atom_mask = node_mask,
                pair_mask = edge_mask,
            )
            
            # Compute weighted L2 loss between predicted x1 and true x1.
            loss = compute_mse_loss(
                denoised_atom_coords=velocity,
                true_atom_coords=vt,
                sigma_loss_weights=1.0,
                batch_reduce="mean",
                return_aligned_coords=False,
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.item())
            
            running_losses["loss"] += loss.item()
            num_batches += 1
            
            # Update tqdm progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())
        
        monitor["loss"].append( running_losses["loss"] / num_batches )
        monitor["lr"].append(optimizer.param_groups[0]["lr"])
        epoch_history.append(epoch+1)
        
        # Plot the loss every plot_interval epochs.
        if (epoch + 1) % plot_interval == 0:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(epoch_history, monitor["loss"], color="b", marker='o')
            ax[0].set_ylabel("Flow Loss")
            # ax[0].set_ylim(0.13, 0.40)
            ax[0].grid(True)

            ax[1].plot(epoch_history, monitor["lr"], color="tomato")
            # Set y-axis to use scientific notation in the form 1e-6, 1e-7, etc.
            ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
            ax[1].set_ylabel("Learning Rate")
            ax[1].grid(True)
            
            # Add shared x-axis label
            fig.supxlabel("Epoch")
            
            # Adjust layout to make room for the shared label
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(f"{FIG_DIR}/{MOLECULE}_{epoch+1}_epoch_loss.png")


    model.eval();
    wrapped_model = InferenceWrapper(network=model)
    adp = ADP.from_pdb(f"{DATASET_DIR}/debug/{MOLECULE}.pdb")
    res = Pipeline.generate_from_flow(
        n_samples=10_000,
        samples_per_batch=5_000,
        n_timesteps=100,
        molecules=[adp],
        flow_model=wrapped_model,
        interpolant=interpolant,
        method="dopri5",
    )
    samples_th = res[MOLECULE]["samples"]


    cfg = initialize_config(config_dir="/net/galaxy/home/koes/dap181/labspace/aita/configs/", overrides=["experiment=anneal_adp"])
    forcefield_partial = hyrda_init("energy", cfg)
    u = forcefield_partial(temperature=1200.00)

    ref_energies = -u(ref_coords, return_force=False)
    gen_energies = -u(samples_th.reshape(-1, 66) / 10.0, return_force=False)
    print(f"Min energy {gen_energies.min()} | Max Energy: {gen_energies.max()}")
    print (f"Number of sample with energy > 60 kJ/mol: {sum(gen_energies > 60.0)}")

    # Plot the energy histograms
    ref_np = ref_energies.detach().cpu().numpy().flatten()
    gen_np = gen_energies.detach().cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(ref_np, bins=100, density=True, alpha=0.55, label="Reference (MD)", color="tomato", edgecolor="white", linewidth=0.4)
    ax.hist(gen_np[gen_np < 60.0], bins=100, density=True, alpha=0.55, label="Generated (Flow)", color="dodgerblue", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Energy (kJ/mol)")
    ax.set_ylabel("Density")
    ax.set_title(f"{MOLECULE} — Energy Distribution")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{MOLECULE}_energy_histogram.png", dpi=200)
    plt.close(fig)

    # Clean up and exit safely
    print("Done!")
    exit(0)

if __name__ == "__main__":
    main()