import torch
import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation
from IPR_MPNN.samplers.simple_scheme import SIMPLESampler

class IPR_MPNN_Conv(nn.Module):
    def __init__(self, n_in, n_selections=64, n_virtual=2, device='cuda'):
        super().__init__()
        self.n_virtual = n_virtual
        self.n_in = n_in
        self.device = device

        # Maps input features to scores for virtual node assignment
        self.feature_reducer = nn.Linear(n_in, n_virtual)

        # Selects n_selections real nodes for each virtual node
        self.sampler = SIMPLESampler(k=n_selections, device=device, n_samples=1)

        # PyG Aggregation modules
        self.agg_r_v = MeanAggregation() # real to virtual
        self.agg_v_v = nn.Sequential(
            nn.Linear(n_in * n_virtual, n_in * n_virtual),
            nn.ReLU(),
            nn.Linear(n_in * n_virtual, n_in * n_virtual)
        ) # virtual to virtual

    def forward(self, x, n_batches):
        scores = self.feature_reducer(x)  # [N_total, n_virtual]
        n_real = x.size(0) // n_batches

        for batch_idx in range(n_batches):
            start = batch_idx * n_real
            end = (batch_idx + 1) * n_real
            x_batch = x[start:end]
            scores_batch = scores[start:end]

            mask, _ = self.sampler(scores_batch.unsqueeze(-1))  # [1, n_real, n_virtual, 1]
            mask = mask.squeeze().bool()  # [n_real, n_virtual]

            # Prepare data for aggregation
            virtual_indices = torch.arange(self.n_virtual, device=self.device)
            virtual_indices = virtual_indices.unsqueeze(0).expand(n_real, -1)  # [n_real, n_virtual]
            
            # Flatten for aggregation
            mask_flat = mask.view(-1)  # [n_real * n_virtual]
            x_expanded = x_batch.unsqueeze(1).expand(-1, self.n_virtual, -1)  # [n_real, n_virtual, n_features]
            x_flat = x_expanded.reshape(-1, self.n_in)  # [n_real * n_virtual, n_features]
            indices_flat = virtual_indices.reshape(-1)  # [n_real * n_virtual]

            # === Step 1: Updating Virtual Nodes === #
            virt_features = self.agg_r_v(
                x_flat[mask_flat],
                index=indices_flat[mask_flat],
                dim_size=self.n_virtual
            )  # [n_virtual, n_features]

            # === Step 2: Updating Among Virtual Nodes === #
            # Flatten all virtual features, process through MLP, then reshape back
            updated_virt = self.agg_v_v(
                virt_features.view(1, -1)  # [1, n_virtual * n_in]
            ).view(self.n_virtual, self.n_in)  # [n_virtual, n_in]

            # === Step 3: Updating Original Nodes === #
            # For each real node, average with all virtual nodes it's connected to
            for v in range(self.n_virtual):
                connected_real = mask[:, v]  # [n_real] bool mask
                x[start:end][connected_real] = (x_batch[connected_real] + updated_virt[v].expand_as(x_batch[connected_real])) / 2

        return x