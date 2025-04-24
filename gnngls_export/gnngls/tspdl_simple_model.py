"""
Simple GNN model for TSPDL.

This module provides a simplified GNN model for TSPDL problems.
"""

import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from typing import Optional


class SimpleGNNModel(nn.Module):
    """
    Simple GNN model for TSPDL.
    
    This model uses a simple GNN architecture to predict edge properties.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2
    ):
        """
        Initialize the model.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            n_layers: Number of GNN layers
        """
        super(SimpleGNNModel, self).__init__()
        
        # Input embedding
        self.edge_embedding = nn.Linear(in_dim, hidden_dim)
        self.node_embedding = nn.Linear(4, hidden_dim)  # [x, y, demand, draft_limit]
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            dglnn.GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(
        self,
        g: dgl.DGLGraph,
        edge_feats: torch.Tensor,
        node_feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            g: DGL graph
            edge_feats: Edge features
            node_feats: Node features (optional)
            
        Returns:
            edge_preds: Edge predictions
        """
        # Embed edge features
        h_e = self.edge_embedding(edge_feats)
        g.edata['h'] = h_e
        
        # Embed node features if provided
        if node_feats is not None:
            h_v = self.node_embedding(node_feats)
            g.ndata['h'] = h_v
        
        # Apply GNN layers
        h = g.ndata['h']
        for layer in self.gnn_layers:
            h = layer(g, h)
            h = torch.relu(h)
        g.ndata['h'] = h
        
        # Get edge embeddings
        g.apply_edges(lambda edges: {'h': edges.src['h'] + edges.dst['h']})
        edge_embeds = g.edata['h']
        
        # Apply output layer
        edge_preds = self.output(edge_embeds)
        
        return edge_preds
