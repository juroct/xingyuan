"""
Visualization utilities for GNNGLS.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import torch
from typing import List, Tuple, Dict, Optional, Union, Any

from .solvers import ANY_SOLVER_AVAILABLE
if ANY_SOLVER_AVAILABLE:
    from .solvers import get_optimal_tour


def plot_tsp_instance(
    G: nx.Graph,
    title: str = "TSP Instance",
    node_size: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a TSP instance.
    
    Args:
        G: NetworkX graph representing the TSP instance
        title: Plot title
        node_size: Size of nodes in the plot
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='skyblue', 
        node_size=node_size,
        ax=ax
    )
    
    # Draw edges with weights as colors
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize weights for coloring
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
        norm = plt.Normalize(min_weight, max_weight)
        cmap = cm.viridis
        
        # Draw edges
        for i, (u, v) in enumerate(edges):
            ax.plot(
                [pos[u][0], pos[v][0]], 
                [pos[u][1], pos[v][1]],
                alpha=0.5,
                color=cmap(norm(weights[i])),
                linewidth=1
            )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Edge Weight')
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_family='sans-serif',
        ax=ax
    )
    
    # Highlight depot
    if 0 in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[0], 
            node_color='red', 
            node_size=node_size*1.5,
            ax=ax
        )
    
    # Set title and axis properties
    ax.set_title(title)
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal')
    
    # Adjust limits to include all nodes with some padding
    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    x_margin = (max(x_values) - min(x_values)) * 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.1
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_tour(
    G: nx.Graph,
    tour: List[int],
    title: str = "TSP Tour",
    node_size: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    highlight_edges: Optional[List[Tuple[int, int]]] = None
) -> plt.Figure:
    """
    Plot a TSP tour.
    
    Args:
        G: NetworkX graph representing the TSP instance
        tour: List of nodes representing the tour
        title: Plot title
        node_size: Size of nodes in the plot
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        highlight_edges: List of edges to highlight
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='skyblue', 
        node_size=node_size,
        ax=ax
    )
    
    # Draw tour edges
    tour_edges = list(zip(tour[:-1], tour[1:]))
    
    # Draw tour edges
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=tour_edges, 
        width=2, 
        alpha=0.8, 
        edge_color='blue',
        ax=ax
    )
    
    # Highlight specific edges if provided
    if highlight_edges:
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=highlight_edges, 
            width=3, 
            alpha=1.0, 
            edge_color='red',
            ax=ax
        )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_family='sans-serif',
        ax=ax
    )
    
    # Highlight depot
    if 0 in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[0], 
            node_color='red', 
            node_size=node_size*1.5,
            ax=ax
        )
    
    # Calculate tour cost
    tour_cost = sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1))
    
    # Set title and axis properties
    ax.set_title(f"{title} (Cost: {tour_cost:.4f})")
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal')
    
    # Add tour order annotations
    for i, node in enumerate(tour[:-1]):  # Skip the last node (same as first)
        ax.annotate(
            f"{i}",
            xy=pos[node],
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )
    
    # Adjust limits to include all nodes with some padding
    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    x_margin = (max(x_values) - min(x_values)) * 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.1
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_algorithm_comparison(
    G: nx.Graph,
    algorithm_tours: Dict[str, List[int]],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a comparison of different algorithms.
    
    Args:
        G: NetworkX graph representing the TSP instance
        algorithm_tours: Dictionary mapping algorithm names to tours
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        
    Returns:
        fig: Matplotlib figure
    """
    n_algorithms = len(algorithm_tours)
    n_cols = min(3, n_algorithms)
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Calculate costs
    costs = {}
    for name, tour in algorithm_tours.items():
        costs[name] = sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1))
    
    # Find optimal tour if solvers are available
    optimal_cost = None
    if ANY_SOLVER_AVAILABLE:
        try:
            coords = [G.nodes[i]['coords'] for i in range(len(G.nodes))]
            _, opt_cost = get_optimal_tour(coords, verbose=False)
            optimal_cost = opt_cost
        except Exception as e:
            print(f"Could not compute optimal tour: {e}")
    
    # Plot each algorithm's tour
    for i, (name, tour) in enumerate(algorithm_tours.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, 
                node_color='skyblue', 
                node_size=50,
                ax=ax
            )
            
            # Draw tour edges
            tour_edges = list(zip(tour[:-1], tour[1:]))
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=tour_edges, 
                width=1.5, 
                alpha=0.8, 
                edge_color='blue',
                ax=ax
            )
            
            # Highlight depot
            if 0 in G.nodes:
                nx.draw_networkx_nodes(
                    G, pos, 
                    nodelist=[0], 
                    node_color='red', 
                    node_size=75,
                    ax=ax
                )
            
            # Set title with cost and gap to optimal if available
            title = f"{name} (Cost: {costs[name]:.4f})"
            if optimal_cost:
                gap = 100 * (costs[name] / optimal_cost - 1)
                title += f", Gap: {gap:.2f}%"
            
            ax.set_title(title)
            ax.set_axis_on()
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_aspect('equal')
    
    # Hide empty subplots
    for i in range(n_algorithms, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    plt.suptitle("Algorithm Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_gnn_predictions(
    G: nx.Graph,
    edge_scores: torch.Tensor,
    tour: Optional[List[int]] = None,
    title: str = "GNN Edge Predictions",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot GNN edge predictions.
    
    Args:
        G: NetworkX graph representing the TSP instance
        edge_scores: Edge scores predicted by the GNN
        tour: Optional tour to overlay
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='skyblue', 
        node_size=100,
        ax=ax
    )
    
    # Draw edges with predicted scores as colors
    edges = list(G.edges())
    
    # Create a mapping from edge to index
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Normalize scores for coloring
    scores = edge_scores.detach().cpu().numpy().flatten()
    norm = plt.Normalize(scores.min(), scores.max())
    cmap = cm.viridis
    
    # Draw edges
    for i, (u, v) in enumerate(edges):
        ax.plot(
            [pos[u][0], pos[v][0]], 
            [pos[u][1], pos[v][1]],
            alpha=0.7,
            color=cmap(norm(scores[i])),
            linewidth=2
        )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Edge Score')
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_family='sans-serif',
        ax=ax
    )
    
    # Highlight depot
    if 0 in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[0], 
            node_color='red', 
            node_size=150,
            ax=ax
        )
    
    # Overlay tour if provided
    if tour:
        tour_edges = list(zip(tour[:-1], tour[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=tour_edges, 
            width=3, 
            alpha=0.8, 
            edge_color='red',
            ax=ax
        )
        
        # Calculate tour cost
        tour_cost = sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1))
        title += f" (Tour Cost: {tour_cost:.4f})"
    
    # Set title and axis properties
    ax.set_title(title)
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal')
    
    # Adjust limits to include all nodes with some padding
    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    x_margin = (max(x_values) - min(x_values)) * 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.1
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_guided_local_search_progress(
    G: nx.Graph,
    tours: List[List[int]],
    costs: List[float],
    penalties: Optional[List[np.ndarray]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True,
    max_tours: int = 4
) -> plt.Figure:
    """
    Plot the progress of guided local search.
    
    Args:
        G: NetworkX graph representing the TSP instance
        tours: List of tours at different iterations
        costs: List of costs at different iterations
        penalties: List of edge penalties at different iterations
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        max_tours: Maximum number of tours to plot
        
    Returns:
        fig: Matplotlib figure
    """
    # Determine layout
    n_tours = min(len(tours), max_tours)
    n_cols = min(2, n_tours)
    n_rows = (n_tours + n_cols - 1) // n_cols + 1  # +1 for the cost plot
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols)
    
    # Plot cost progress
    ax_cost = fig.add_subplot(gs[0, :])
    ax_cost.plot(costs, 'b-', marker='o')
    ax_cost.set_title('Cost Progress')
    ax_cost.set_xlabel('Iteration')
    ax_cost.set_ylabel('Cost')
    ax_cost.grid(True)
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Select tours to plot (first, some middle ones, and last)
    if len(tours) <= max_tours:
        tour_indices = list(range(len(tours)))
    else:
        # Always include first and last
        middle_indices = np.linspace(1, len(tours) - 2, max_tours - 2, dtype=int)
        tour_indices = [0] + list(middle_indices) + [len(tours) - 1]
    
    # Plot selected tours
    for i, idx in enumerate(tour_indices):
        row = (i // n_cols) + 1  # +1 because first row is for cost plot
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='skyblue', 
            node_size=50,
            ax=ax
        )
        
        # Draw tour edges
        tour = tours[idx]
        tour_edges = list(zip(tour[:-1], tour[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=tour_edges, 
            width=1.5, 
            alpha=0.8, 
            edge_color='blue',
            ax=ax
        )
        
        # If penalties are provided, highlight high-penalty edges
        if penalties is not None and idx < len(penalties):
            penalty = penalties[idx]
            edges = list(G.edges())
            
            # Find edges with high penalties
            if np.any(penalty > 0):
                high_penalty_threshold = np.percentile(penalty[penalty > 0], 75)
                high_penalty_edges = [edges[i] for i, p in enumerate(penalty) if p >= high_penalty_threshold]
                
                # Draw high-penalty edges
                nx.draw_networkx_edges(
                    G, pos, 
                    edgelist=high_penalty_edges, 
                    width=2, 
                    alpha=0.6, 
                    edge_color='red',
                    style='dashed',
                    ax=ax
                )
        
        # Highlight depot
        if 0 in G.nodes:
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[0], 
                node_color='red', 
                node_size=75,
                ax=ax
            )
        
        # Set title
        ax.set_title(f"Iteration {idx} (Cost: {costs[idx]:.4f})")
        ax.set_axis_on()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect('equal')
    
    # Add overall title
    plt.suptitle("Guided Local Search Progress", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_animation(
    G: nx.Graph,
    tours: List[List[int]],
    costs: List[float],
    output_path: str,
    title: str = "Algorithm Progress",
    fps: int = 2
):
    """
    Create an animation of algorithm progress.
    
    Args:
        G: NetworkX graph representing the TSP instance
        tours: List of tours at different iterations
        costs: List of costs at different iterations
        output_path: Path to save the animation
        title: Animation title
        fps: Frames per second
    """
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Could not import matplotlib.animation. Please install it with 'pip install matplotlib'.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Initialize plot
    def init():
        ax.clear()
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='skyblue', 
            node_size=100,
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos, 
            font_size=10, 
            font_family='sans-serif',
            ax=ax
        )
        
        # Highlight depot
        if 0 in G.nodes:
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[0], 
                node_color='red', 
                node_size=150,
                ax=ax
            )
        
        # Set axis properties
        ax.set_axis_on()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect('equal')
        
        # Adjust limits to include all nodes with some padding
        x_values = [x for x, y in pos.values()]
        y_values = [y for x, y in pos.values()]
        x_margin = (max(x_values) - min(x_values)) * 0.1
        y_margin = (max(y_values) - min(y_values)) * 0.1
        ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
        ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
        
        return []
    
    # Update function for animation
    def update(frame):
        ax.clear()
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='skyblue', 
            node_size=100,
            ax=ax
        )
        
        # Draw tour edges
        tour = tours[frame]
        tour_edges = list(zip(tour[:-1], tour[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=tour_edges, 
            width=2, 
            alpha=0.8, 
            edge_color='blue',
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos, 
            font_size=10, 
            font_family='sans-serif',
            ax=ax
        )
        
        # Highlight depot
        if 0 in G.nodes:
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[0], 
                node_color='red', 
                node_size=150,
                ax=ax
            )
        
        # Set title with iteration and cost
        ax.set_title(f"{title} - Iteration {frame} (Cost: {costs[frame]:.4f})")
        
        # Set axis properties
        ax.set_axis_on()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect('equal')
        
        # Adjust limits to include all nodes with some padding
        x_values = [x for x, y in pos.values()]
        y_values = [y for x, y in pos.values()]
        x_margin = (max(x_values) - min(x_values)) * 0.1
        y_margin = (max(y_values) - min(y_values)) * 0.1
        ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
        ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
        
        return []
    
    # Create animation
    ani = FuncAnimation(
        fig, update, frames=len(tours),
        init_func=init, blit=True, repeat=True
    )
    
    # Save animation
    ani.save(output_path, writer='pillow', fps=fps)
    
    plt.close()
    
    print(f"Animation saved to {output_path}")
