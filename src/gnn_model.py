"""Plain PyTorch graph neural network for graph-level regression."""

from __future__ import annotations

import torch
from torch import nn


class GraphConvLayer(nn.Module):
    """A small GCN-style layer using sum-normalized neighbor aggregation."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.self_linear = nn.Linear(in_channels, out_channels)
        self.neigh_linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        agg = torch.zeros_like(x)
        if edge_index.numel() > 0:
            agg.index_add_(0, row, x[col])
            deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            deg.index_add_(0, row, torch.ones_like(row, dtype=x.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(1)

        h = self.self_linear(x) + self.neigh_linear(agg)
        h = torch.relu(self.norm(h))
        return self.dropout(h)


class EdgeGatedGraphConvLayer(nn.Module):
    """Message passing layer that conditions neighbor messages on bond features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_linear = nn.Linear(in_channels, out_channels)
        self.neigh_linear = nn.Linear(in_channels, out_channels)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_channels, out_channels),
            nn.Sigmoid(),
        )
        self.edge_bias = nn.Linear(edge_channels, out_channels, bias=False)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels, bias=False)
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row, col = edge_index
        agg = torch.zeros(x.size(0), self.neigh_linear.out_features, device=x.device, dtype=x.dtype)
        if edge_index.numel() > 0:
            messages = self.neigh_linear(x[col])
            if edge_attr is not None and edge_attr.numel() > 0:
                edge_attr = edge_attr.to(dtype=x.dtype)
                messages = messages * self.edge_gate(edge_attr) + self.edge_bias(edge_attr)
            agg.index_add_(0, row, messages)
            deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            deg.index_add_(0, row, torch.ones_like(row, dtype=x.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(1)

        h = self.self_linear(x) + agg
        h = torch.relu(self.norm(h))
        return self.dropout(h) + self.residual(x)


class GINEGraphConvLayer(nn.Module):
    """GIN-style edge-aware layer using sum aggregation.

    Sum aggregation follows the 1-WL graph refinement view more closely than
    mean aggregation, preserving neighbor multiplicities that can matter for
    molecular graph topology.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_encoder = (
            nn.Linear(edge_channels, in_channels)
            if edge_channels > 0
            else None
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels, bias=False)
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row, col = edge_index
        agg = torch.zeros_like(x)
        if edge_index.numel() > 0:
            messages = x[col]
            if self.edge_encoder is not None and edge_attr is not None and edge_attr.numel() > 0:
                messages = messages + self.edge_encoder(edge_attr.to(dtype=x.dtype))
            messages = torch.relu(messages)
            agg.index_add_(0, row, messages)

        h = (1.0 + self.eps) * x + agg
        h = torch.relu(self.norm(self.mlp(h)))
        return self.dropout(h) + self.residual(x)


class MolecularGNN(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_outputs: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        graph_feature_dim: int = 0,
        use_graph_feature_encoder: bool = False,
        num_edge_features: int = 0,
        message_passing: str = "gcn",
        readout: str = "mean_max",
        jk_mode: str = "last",
    ):
        super().__init__()
        self.graph_feature_dim = graph_feature_dim
        self.use_graph_feature_encoder = use_graph_feature_encoder and graph_feature_dim > 0
        self.message_passing = message_passing
        self.readout_mode = readout
        self.jk_mode = jk_mode
        layers: list[nn.Module] = []
        in_dim = num_node_features
        for _ in range(num_layers):
            if message_passing == "gcn":
                layers.append(GraphConvLayer(in_dim, hidden_dim, dropout))
            elif message_passing == "edge_gated":
                layers.append(
                    EdgeGatedGraphConvLayer(
                        in_channels=in_dim,
                        out_channels=hidden_dim,
                        edge_channels=num_edge_features,
                        dropout=dropout,
                    )
                )
            elif message_passing == "gine":
                layers.append(
                    GINEGraphConvLayer(
                        in_channels=in_dim,
                        out_channels=hidden_dim,
                        edge_channels=num_edge_features,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(f"Unknown message passing layer: {message_passing}")
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        if jk_mode == "last":
            node_embedding_dim = hidden_dim
        elif jk_mode == "concat":
            node_embedding_dim = hidden_dim * num_layers
        else:
            raise ValueError(f"Unknown JK mode: {jk_mode}")
        if readout == "mean_max":
            pooled_dim = node_embedding_dim * 2
            self.attention_gate = None
        elif readout == "attention_mean_max":
            pooled_dim = node_embedding_dim * 3
            self.attention_gate = nn.Linear(node_embedding_dim, 1)
        else:
            raise ValueError(f"Unknown readout: {readout}")
        if self.use_graph_feature_encoder:
            self.graph_feature_encoder = nn.Sequential(
                nn.Linear(graph_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            readout_in_dim = pooled_dim + hidden_dim
        else:
            self.graph_feature_encoder = None
            readout_in_dim = pooled_dim + graph_feature_dim

        self.readout = nn.Sequential(
            nn.Linear(readout_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_outputs: list[torch.Tensor] = []
        for layer in self.layers:
            if self.message_passing in {"edge_gated", "gine"}:
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)
            layer_outputs.append(x)
        if self.jk_mode == "concat":
            x = torch.cat(layer_outputs, dim=1)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        sum_pool = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        sum_pool.index_add_(0, batch, x)
        counts = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        mean_pool = sum_pool / counts.clamp_min(1.0).unsqueeze(1)

        max_pool = torch.full_like(mean_pool, -1.0e9)
        for graph_idx in range(num_graphs):
            mask = batch == graph_idx
            max_pool[graph_idx] = x[mask].max(dim=0).values

        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)
        if self.attention_gate is not None:
            attn_pool = torch.zeros_like(mean_pool)
            logits = self.attention_gate(x).squeeze(1)
            for graph_idx in range(num_graphs):
                mask = batch == graph_idx
                weights = torch.softmax(logits[mask], dim=0).unsqueeze(1)
                attn_pool[graph_idx] = (weights * x[mask]).sum(dim=0)
            graph_embedding = torch.cat([graph_embedding, attn_pool], dim=1)
        if self.graph_feature_dim:
            if graph_features is None:
                graph_features = torch.zeros(
                    num_graphs,
                    self.graph_feature_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
            if self.graph_feature_encoder is not None:
                graph_features = self.graph_feature_encoder(graph_features)
            graph_embedding = torch.cat([graph_embedding, graph_features], dim=1)
        return self.readout(graph_embedding)


def collate_graphs(graphs: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    graph_feature_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    edge_attr_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    smiles: list[str] = []
    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        x = graph["x"]
        x_parts.append(x)
        if "y" in graph:
            y_parts.append(graph["y"].unsqueeze(0))
        if "graph_features" in graph:
            graph_feature_parts.append(graph["graph_features"].unsqueeze(0))
        if graph["edge_index"].numel() > 0:
            edge_parts.append(graph["edge_index"] + node_offset)
            if "edge_attr" in graph:
                edge_attr_parts.append(graph["edge_attr"])
        batch_parts.append(torch.full((x.size(0),), graph_idx, dtype=torch.long))
        smiles.append(graph["smiles"])
        node_offset += x.size(0)

    edge_index = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long)
    )
    batch = {
        "x": torch.cat(x_parts, dim=0),
        "edge_index": edge_index,
        "batch": torch.cat(batch_parts, dim=0),
        "smiles": smiles,
    }
    if y_parts:
        batch["y"] = torch.cat(y_parts, dim=0)
    if graph_feature_parts:
        batch["graph_features"] = torch.cat(graph_feature_parts, dim=0)
    if edge_attr_parts:
        batch["edge_attr"] = torch.cat(edge_attr_parts, dim=0)
    return batch
