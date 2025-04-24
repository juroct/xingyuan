#!/usr/bin/env python
# coding: utf-8

"""
Extract model state dict from checkpoint.
"""

import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model state dict from checkpoint")
    parser.add_argument("input_path", type=str, help="Input checkpoint path")
    parser.add_argument("output_path", type=str, help="Output model path")
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.input_path)
    
    # Extract model state dict
    model_state_dict = checkpoint["model_state_dict"]
    
    # Save model state dict
    torch.save(model_state_dict, args.output_path)
    
    print(f"Model state dict extracted from {args.input_path} and saved to {args.output_path}")
