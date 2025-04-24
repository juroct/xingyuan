#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import pathlib
import time

import dgl
import networkx as nx
import numpy as np
import torch
import tqdm.auto as tqdm
from torch.utils.data import DataLoader

from gnngls import models, datasets, algorithms, tour_cost


def evaluate_model(model, data_loader, device):
    """Evaluate the model on the test set."""
    model.eval()
    
    results = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device)
            x = batch.edata['features']
            
            # Get model predictions
            y_pred = model(batch, x)
            
            # Process each graph in the batch
            for g_idx in range(batch.batch_size):
                g = dgl.unbatch(batch)[g_idx]
                
                # Get the predictions for this graph
                g_edges = g.edges()
                g_pred = y_pred[g.batch_num_edges() * g_idx:g.batch_num_edges() * (g_idx + 1)]
                
                # Convert to networkx for algorithms
                nx_g = dgl.to_networkx(g, edge_attrs=['weight'])
                
                # Add the predictions as edge attributes
                for i, (src, dst) in enumerate(zip(g_edges[0], g_edges[1])):
                    nx_g.edges[src.item(), dst.item()]['pred'] = g_pred[i].item()
                
                # Run algorithms
                depot = 0  # Assuming node 0 is the depot
                
                # Nearest neighbor
                start_time = time.time()
                nn_tour = algorithms.nearest_neighbor(nx_g, depot)
                nn_time = time.time() - start_time
                nn_cost = tour_cost(nx_g, nn_tour)
                
                # Guided local search with weight
                start_time = time.time()
                t_limit = start_time + args.time_limit
                gls_weight_tour, gls_weight_cost, _ = algorithms.guided_local_search(
                    nx_g, nn_tour, nn_cost, t_limit, guides=['weight']
                )
                gls_weight_time = time.time() - start_time
                
                # Guided local search with model predictions
                start_time = time.time()
                t_limit = start_time + args.time_limit
                gls_pred_tour, gls_pred_cost, _ = algorithms.guided_local_search(
                    nx_g, nn_tour, nn_cost, t_limit, guides=['pred']
                )
                gls_pred_time = time.time() - start_time
                
                # Store results
                results.append({
                    'nn_cost': nn_cost,
                    'nn_time': nn_time,
                    'gls_weight_cost': gls_weight_cost,
                    'gls_weight_time': gls_weight_time,
                    'gls_pred_cost': gls_pred_cost,
                    'gls_pred_time': gls_pred_time,
                })
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('model_path', type=pathlib.Path, help='Path to the trained model')
    parser.add_argument('data_path', type=pathlib.Path, help='Path to the test data')
    parser.add_argument('output_path', type=pathlib.Path, help='Path to save results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--time_limit', type=float, default=10.0, help='Time limit for GLS in seconds')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()
    
    # Load test data
    test_set = datasets.TSPDataset(args.data_path)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=dgl.batch)
    
    # Load model
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get model parameters from the first graph
    _, feat_dim = test_set[0].edata['features'].shape
    
    # Create model with the same architecture
    model = models.EdgePropertyPredictionModel(
        feat_dim,
        128,  # embed_dim, should match the trained model
        1,    # out_dim
        3,    # n_layers, should match the trained model
        n_heads=8,  # should match the trained model
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    nn_costs = [r['nn_cost'] for r in results]
    gls_weight_costs = [r['gls_weight_cost'] for r in results]
    gls_pred_costs = [r['gls_pred_cost'] for r in results]
    
    print(f"Average NN cost: {np.mean(nn_costs):.2f}")
    print(f"Average GLS (weight) cost: {np.mean(gls_weight_costs):.2f}")
    print(f"Average GLS (pred) cost: {np.mean(gls_pred_costs):.2f}")
    
    print(f"Improvement of GLS (weight) over NN: {100 * (1 - np.mean(gls_weight_costs) / np.mean(nn_costs)):.2f}%")
    print(f"Improvement of GLS (pred) over NN: {100 * (1 - np.mean(gls_pred_costs) / np.mean(nn_costs)):.2f}%")
    print(f"Improvement of GLS (pred) over GLS (weight): {100 * (1 - np.mean(gls_pred_costs) / np.mean(gls_weight_costs)):.2f}%")
