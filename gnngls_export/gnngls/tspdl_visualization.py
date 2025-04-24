"""
Visualization utilities for TSPDL (Traveling Salesperson Problem with Draft Limits).

This module provides functions for visualizing TSPDL instances and solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union, Any

from .tspdl import TSPDLInstance, TSPDLSolution
from .visualization import plot_tour


def plot_tspdl_instance(
    instance: TSPDLInstance,
    title: str = "TSPDL Instance",
    node_size: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a TSPDL instance.

    Args:
        instance: TSPDL instance
        title: Plot title
        node_size: Size of nodes in the plot
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure

    Returns:
        fig: Matplotlib figure
    """
    # Convert instance to NetworkX graph
    G = instance.to_networkx()

    fig, ax = plt.subplots(figsize=figsize)

    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')

    # Get node demands and draft limits
    demands = nx.get_node_attributes(G, 'demand')
    draft_limits = nx.get_node_attributes(G, 'draft_limit')

    # Normalize demands and draft limits for coloring
    max_demand = max(demands.values())
    min_demand = min(demands.values())

    max_draft_limit = max(draft_limits.values())
    min_draft_limit = min(draft_limits.values())

    # Create node colors based on demand
    node_colors = []
    for node in G.nodes():
        # Normalize demand to [0, 1]
        if max_demand > min_demand:
            norm_demand = (demands[node] - min_demand) / (max_demand - min_demand)
        else:
            norm_demand = 0

        # Use a colormap
        node_colors.append(plt.cm.viridis(norm_demand))

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
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

        # Add colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Edge Weight')

    # Draw node labels with demand and draft limit
    labels = {}
    for node in G.nodes():
        labels[node] = f"{node}\nD:{demands[node]:.2f}\nL:{draft_limits[node]:.2f}"

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
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

    # Add legend for node colors
    if max_demand > min_demand:
        # Create a colormap for the legend
        cmap = plt.cm.viridis
        norm = plt.Normalize(min_demand, max_demand)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add colorbar for node demands
        cbar = plt.colorbar(sm, ax=ax, location='right', label='Demand')

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


def plot_tspdl_solution(
    solution: TSPDLSolution,
    title: str = "TSPDL Solution",
    node_size: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a TSPDL solution.

    Args:
        solution: TSPDL solution
        title: Plot title
        node_size: Size of nodes in the plot
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure

    Returns:
        fig: Matplotlib figure
    """
    # Convert solution to NetworkX graph
    G = solution.to_networkx()

    fig, ax = plt.subplots(figsize=figsize)

    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')

    # Get node demands and draft limits
    demands = nx.get_node_attributes(G, 'demand')
    draft_limits = nx.get_node_attributes(G, 'draft_limit')

    # Get node loads if available
    loads = nx.get_node_attributes(G, 'load')

    # Get tour edges
    tour_edges = []
    for i, j in G.edges():
        if G.edges[i, j].get('in_tour', False):
            tour_edges.append((i, j))

    # Get edges with excess draft limit
    excess_edges = []
    for i, j in tour_edges:
        if G.edges[i, j].get('excess', 0) > 0:
            excess_edges.append((i, j))

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='skyblue',
        node_size=node_size,
        ax=ax
    )

    # Draw tour edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=tour_edges,
        width=2,
        alpha=0.8,
        edge_color='blue',
        ax=ax
    )

    # Draw excess edges
    if excess_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=excess_edges,
            width=3,
            alpha=1.0,
            edge_color='red',
            style='dashed',
            ax=ax
        )

    # Draw node labels with demand, draft limit, and load
    labels = {}
    for node in G.nodes():
        label = f"{node}\nD:{demands[node]:.2f}\nL:{draft_limits[node]:.2f}"
        if node in loads:
            label += f"\nLoad:{loads[node]:.2f}"
        labels[node] = label

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
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

    # Add visit order annotations
    visit_orders = nx.get_node_attributes(G, 'visit_order')
    for node, order in visit_orders.items():
        ax.annotate(
            f"{order}",
            xy=pos[node],
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )

    # Set title with cost and feasibility
    title_text = f"{title} (Cost: {solution.cost:.4f})"
    if not solution.feasible:
        title_text += " [INFEASIBLE]"
    ax.set_title(title_text)

    # Set axis properties
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


def plot_tspdl_algorithm_comparison(
    instance: TSPDLInstance,
    algorithm_tours: Dict[str, List[int]],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a comparison of different algorithms for TSPDL.

    Args:
        instance: TSPDL instance
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

    # Convert instance to NetworkX graph
    G = instance.to_networkx()

    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')

    # Calculate costs and feasibility
    costs = {}
    feasible = {}
    for name, tour in algorithm_tours.items():
        solution = TSPDLSolution(instance, tour)
        costs[name] = solution.cost
        feasible[name] = solution.feasible

    # Plot each algorithm's tour
    for i, (name, tour) in enumerate(algorithm_tours.items()):
        if i < len(axes):
            ax = axes[i]

            # Create solution
            solution = TSPDLSolution(instance, tour)

            # Get tour edges
            tour_edges = list(zip(tour[:-1], tour[1:]))

            # Get edges with excess draft limit
            excess_edges = []
            for j, edge in enumerate(tour_edges):
                if j < len(solution.out_of_draft_limit) and solution.out_of_draft_limit[j] > 0:
                    excess_edges.append(edge)

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_color='skyblue',
                node_size=50,
                ax=ax
            )

            # Draw tour edges
            nx.draw_networkx_edges(
                G, pos,
                edgelist=tour_edges,
                width=1.5,
                alpha=0.8,
                edge_color='blue',
                ax=ax
            )

            # Draw excess edges
            if excess_edges:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=excess_edges,
                    width=2,
                    alpha=1.0,
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

            # Set title with cost and feasibility
            title = f"{name} (Cost: {costs[name]:.4f})"
            if not feasible[name]:
                title += " [INFEASIBLE]"

            ax.set_title(title)
            ax.set_axis_on()
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_aspect('equal')

    # Hide empty subplots
    for i in range(n_algorithms, len(axes)):
        axes[i].axis('off')

    # Add overall title
    plt.suptitle("TSPDL Algorithm Comparison", fontsize=16)
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


def plot_tspdl_gls_progress(
    instance: TSPDLInstance,
    tours: List[List[int]],
    costs: List[float],
    out_of_draft_limits: List[float],
    infeasible_nodes: List[int],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True,
    max_tours: int = 4
) -> plt.Figure:
    """
    Plot the progress of guided local search for TSPDL.

    Args:
        instance: TSPDL instance
        tours: List of tours at different iterations
        costs: List of costs at different iterations
        out_of_draft_limits: List of total excess over draft limits at different iterations
        infeasible_nodes: List of number of infeasible nodes at different iterations
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
    n_rows = (n_tours + n_cols - 1) // n_cols + 2  # +2 for the cost and infeasibility plots

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols)

    # Plot cost progress
    ax_cost = fig.add_subplot(gs[0, :])
    ax_cost.plot(costs, 'b-', marker='o')
    ax_cost.set_title('Cost Progress')
    ax_cost.set_xlabel('Iteration')
    ax_cost.set_ylabel('Cost')
    ax_cost.grid(True)

    # Plot infeasibility progress
    ax_infeas = fig.add_subplot(gs[1, :])
    ax_infeas.plot(out_of_draft_limits, 'r-', marker='o', label='Total Excess')
    ax_infeas.plot(infeasible_nodes, 'g-', marker='s', label='Infeasible Nodes')
    ax_infeas.set_title('Infeasibility Progress')
    ax_infeas.set_xlabel('Iteration')
    ax_infeas.set_ylabel('Value')
    ax_infeas.legend()
    ax_infeas.grid(True)

    # Convert instance to NetworkX graph
    G = instance.to_networkx()

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
        row = (i // n_cols) + 2  # +2 because first two rows are for cost and infeasibility plots
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Create solution
        tour = tours[idx]
        solution = TSPDLSolution(instance, tour)

        # Get tour edges
        tour_edges = list(zip(tour[:-1], tour[1:]))

        # Get edges with excess draft limit
        excess_edges = []
        for j, edge in enumerate(tour_edges):
            if j < len(solution.out_of_draft_limit) and solution.out_of_draft_limit[j] > 0:
                excess_edges.append(edge)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='skyblue',
            node_size=50,
            ax=ax
        )

        # Draw tour edges
        nx.draw_networkx_edges(
            G, pos,
            edgelist=tour_edges,
            width=1.5,
            alpha=0.8,
            edge_color='blue',
            ax=ax
        )

        # Draw excess edges
        if excess_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=excess_edges,
                width=2,
                alpha=1.0,
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
        title = f"Iteration {idx} (Cost: {costs[idx]:.4f})"
        if out_of_draft_limits[idx] > 0:
            title += f" [Excess: {out_of_draft_limits[idx]:.2f}]"
        ax.set_title(title)
        ax.set_axis_on()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect('equal')

    # Add overall title
    plt.suptitle("TSPDL Guided Local Search Progress", fontsize=16)
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