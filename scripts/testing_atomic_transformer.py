import dgl
import torch
import numpy as np
import rootutils
import mdtraj as md
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from lightning import seed_everything
from typing import Tuple, Dict, Optional, List

rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

from aita.data.molecule import ADP
from aita.pipeline.pipeline import Pipeline
from aita.plans import TrigPlan
from aita.interpolants import Interpolant
from aita.utils.interactive import view_mol_3d
from aita.data.datasets import SimulationDataset
from aita.utils.rigid_align_loss import compute_mse_loss
from aita.models.atomic_transformer import AtomicTransformer
from aita.models.components.ema import EMA

from aita.utils.random_rotations import random_rotations
from aita.utils.graph_utils import (
    get_batch_indices,
    nodes_to_padded_tensor,
    edges_to_pair_tensor,
    rbf,
)

from aita.utils.interactive import (
    initialize_config,
    hyrda_init,
    view_mol_3d,
)


MOLECULE = "alanine_dipeptide"
FIG_DIR = "/net/galaxy/home/koes/dap181/labspace/aita/scripts/"
DATASET_DIR = "/net/galaxy/home/koes/dap181/labspace/aita/data/"


class LegacyGraphAdapter(object):

    def __init__(self, g: dgl.DGLGraph):
        super(LegacyGraphAdapter, self).__init__()
        self.batch_size = g.batch_size
        self.num_nodes_per_graph = g.batch_num_nodes()
        self.num_edges_per_graph = g.batch_num_edges()
        self.batch_ids_nodes = get_batch_indices(g, data_type="node")
        self.batch_ids_edges = get_batch_indices(g, data_type="edge")
        self.device = g.device
        self.edges = g.edges()
    
    @classmethod
    def adapt_and_pad(cls, g: dgl.DGLGraph, *args, **kwargs) -> Tuple["LegacyGraphAdapter", Dict[str, torch.Tensor]]:
        adapter = cls(g)
        padded = adapter.graph_to_padded_tensor(g, *args, **kwargs)
        return adapter, padded

    @torch.no_grad()
    def apply_random_rotations(self, g: dgl.DGLGraph, node_keys: Optional[List[str]] = None) -> dgl.DGLGraph:
        """
        Apply random rotations to the coordinates of the graph.

        Generates one random SO(3) rotation per molecule in the batch and
        applies it to every 3-D vector node feature (any ndata with shape
        (total_atoms, 3)), e.g. noisy coordinates ``xt``.
        """
        if node_keys is None:
            node_keys = ["xt"]

        coord_keys = [k for k in node_keys if k in g.ndata and g.ndata[k].dim() == 2 and g.ndata[k].size(-1) == 3]
        if not coord_keys:
            return g

        device = g.device
        dtype = g.ndata[coord_keys[0]].dtype
        R = random_rotations(self.batch_size, dtype=dtype, device=device)
        R_per_atom = R[self.batch_ids_nodes.to(device)]

        for key in coord_keys:
            g.ndata[key] = torch.einsum('nd,nds->ns', g.ndata[key], R_per_atom)

        return g

    def compute_rbf_edge_features(
        self,
        padded_coords: torch.Tensor,
        D_min: float = 0.,
        D_max: float = 20.,
        D_count: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Build RBF-encoded pairwise distance features and displacement vectors
        from an already-padded coordinate tensor.

        Args:
            padded_coords: Padded node coordinates, shape
                ``(batch_size, max_num_nodes, 3)``.
            D_min: Minimum distance for RBF centres.
            D_max: Maximum distance for RBF centres.
            D_count: Number of RBF basis functions.

        Returns:
            rbf_features: Dense pairwise RBF features, shape
                ``(batch_size, max_num_nodes, max_num_nodes, D_count)``.
                Padded positions are zeroed out.
            displacements: Unit-length pairwise displacement vectors
                ``(x_j - x_i) / ||x_j - x_i||``, shape
                ``(batch_size, max_num_nodes, max_num_nodes, 3)``.
                Padded positions are zeroed out.
            pair_mask: Boolean mask, shape
                ``(batch_size, max_num_nodes, max_num_nodes)``.
                ``True`` where either the row or column index falls outside
                the molecule's real node count (i.e. padding).
        """
        # (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N, 3)
        displacements = padded_coords.unsqueeze(2) - padded_coords.unsqueeze(1)
        distances = displacements.norm(dim=-1)  # (B, N, N)
        displacements = displacements / (distances.unsqueeze(-1) + 1e-8)

        rbf_features = rbf(distances, D_min=D_min, D_max=D_max, D_count=D_count)

        max_n = padded_coords.size(1)
        node_pad = (
            torch.arange(max_n, device=self.device).unsqueeze(0)
            >= self.num_nodes_per_graph.to(self.device).unsqueeze(1)
        )
        pair_mask = node_pad.unsqueeze(2) | node_pad.unsqueeze(1)
        pair_mask = pair_mask | torch.eye(max_n, device=self.device, dtype=torch.bool).unsqueeze(0)
        rbf_features = rbf_features.masked_fill(pair_mask.unsqueeze(-1), 0.0)
        displacements = displacements.masked_fill(pair_mask.unsqueeze(-1), 0.0)

        return rbf_features, displacements, pair_mask

    @torch.no_grad()
    def graph_to_padded_tensor(
        self,
        g: dgl.DGLGraph,
        coord_key: str = "xt",
        target_key: Optional[str] = "vt",
        feat_key_nodes: str = "attr",
        feat_key_edges: str = "attr",
        D_min: float = 0.,
        D_max: float = 20.,
        D_count: int = 16,
        use_rbf: bool = False,
        return_sigma_t: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Pad node and edge features from a batched DGLGraph into dense tensors,
        appending RBF-encoded pairwise distances to the edge features.

        Args:
            g: Batched DGLGraph.
            coord_key: Key in ``g.ndata`` for 3-D coordinates used by the
                RBF distance encoding.
            target_key: Key in ``g.ndata`` for the regression targets.
                Set to ``None`` at inference time to omit
                ``padded_targets`` from the returned tuple.
            feat_key_nodes: Key in ``g.ndata`` for the node features.
            feat_key_edges: Key in ``g.edata`` for the edge features.
            D_min: Minimum distance for RBF centres.
            D_max: Maximum distance for RBF centres.
            D_count: Number of RBF basis functions.
            use_rbf: If ``True`` (default), append RBF distance features and
                displacement vectors to the edge features.  When ``False``,
                the edge tensor keeps its original feature dimension.
            return_sigma_t: If ``True``, also return the padded noise scale
                ``sigma_t`` from ``g.ndata['sigma_t']``.
        Returns:
            padded_targets: ``(batch_size, N_max, 3)`` — regression targets
                from ``g.ndata[target_key]`` (omitted when
                ``target_key is None``).
            padded_times: ``(batch_size, N_max, 1)`` — diffusion times from
                ``g.ndata['t']``.
            padded_atom_index: ``(batch_size, N_max, 1)`` — per-atom type
                indices from ``g.ndata['atom_index']``.
            padded_nodes: ``(batch_size, N_max, node_feat_dim)``.
            padded_edges: ``(batch_size, N_max, N_max, edge_feat_dim [+ D_count + 3])``
                — when ``use_rbf=True`` the RBF distance features and
                displacement vectors are concatenated along the last axis.
            node_mask: ``(batch_size, N_max)`` — ``True`` at padding positions.
            pair_mask: ``(batch_size, N_max, N_max)`` — ``True`` at padding
                positions.
            padded_sigma_t: ``(batch_size, N_max, 1)`` — noise scale
                (only included when ``return_sigma_t=True``).
        """
        node_feats = g.ndata[feat_key_nodes]  # (total_nodes, node_feat_dim)
        max_n = int(self.num_nodes_per_graph.max().item())

        # Scatter node features into a dense (batch_size, N_max, node_feat_dim)
        # tensor. `offsets` gives the starting global index of each graph's
        # nodes, and `local_idx` converts global node indices to positions
        # within each graph (0 .. n_i-1).
        padded_nodes = node_feats.new_zeros(
            self.batch_size, max_n, node_feats.size(-1)
        )
        offsets = torch.cumsum(self.num_nodes_per_graph, dim=0) - self.num_nodes_per_graph
        local_idx = (
            torch.arange(node_feats.size(0), device=self.device)
            - offsets[self.batch_ids_nodes]
        )
        padded_nodes[self.batch_ids_nodes, local_idx] = node_feats

        # Pad per-node atom type indices into (batch_size, N_max, 1).
        atom_index = g.ndata["atom_index"]
        padded_atom_index = atom_index.new_zeros(self.batch_size, max_n, 1)
        padded_atom_index[self.batch_ids_nodes, local_idx] = atom_index.unsqueeze(-1)

        # Boolean mask marking padding positions as True so that attention
        # layers can ignore them. Shape: (batch_size, N_max).
        node_mask = (
            torch.arange(max_n, device=self.device).unsqueeze(0)
            >= self.num_nodes_per_graph.unsqueeze(1)
        )

        # Convert sparse edge features to a dense pair tensor
        # (batch_size, N_max, N_max, edge_feat_dim).
        padded_edges, pair_mask = edges_to_pair_tensor(g.edata[feat_key_edges], g)

        padded_coords = nodes_to_padded_tensor(g.ndata[coord_key], g)

        if use_rbf:
            rbf_feats, displacements, rbf_mask = self.compute_rbf_edge_features(
                padded_coords, D_min=D_min, D_max=D_max, D_count=D_count,
            )
            pair_mask = rbf_mask
            padded_edges = torch.cat([padded_edges, rbf_feats, displacements], dim=-1)

        # pair_mask from edges_to_pair_tensor only covers padding; add diagonal
        diag = torch.eye(max_n, device=self.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask | diag

        if target_key is not None:
            padded_targets = nodes_to_padded_tensor(g.ndata[target_key], g)

        # Pad per-node flow / diffusion times into (batch_size, N_max, 1), reusing the
        # same local_idx mapping computed above for the node features.
        times = g.ndata["t"]
        padded_times = times.new_zeros(self.batch_size, max_n, 1)
        padded_times[self.batch_ids_nodes, local_idx] = times

        # NOTE: the MASK TENSORS for both nodes and edges indicate
        #       padding positions as FALSE and non-padding positions as TRUE
        result = (padded_times, padded_coords, padded_nodes, padded_atom_index, padded_edges, ~node_mask, ~pair_mask)
        if target_key is not None:
            result = (padded_targets,) + result

        if return_sigma_t:
            # Pad per-node noise scale into (batch_size, N_max, 1).
            sigma_t = g.ndata["sigma_t"]
            padded_sigma_t = sigma_t.new_zeros(self.batch_size, max_n, sigma_t.size(-1))
            padded_sigma_t[self.batch_ids_nodes, local_idx] = sigma_t
            result = result + (padded_sigma_t,)

        return result


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
            self.graph_adapter, *data = LegacyGraphAdapter.adapt_and_pad(g, target_key=None, apply_random_rotations=False, use_rbf=True)
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
        n_layers = 8,
        dropout_prob = 0.1,
        bias = False,
        initial_norm = True,
    )
    model = model.to("cuda:0")
    ema = EMA(model, beta=0.999, update_every=10, allow_different_devices=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    n_epochs = 100
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
            adapter, *data = LegacyGraphAdapter.adapt_and_pad(g, apply_random_rotations=True, use_rbf=True)
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
            ema.update()
            
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