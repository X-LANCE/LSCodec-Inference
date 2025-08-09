# MIT License
# Copyright (c) 2025 Yiwei Guo

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom class for fixed sliding-window average
class SlidingWindowAverage(nn.Module):
    def __init__(self, window_size: int, num_channels: int):
        super(SlidingWindowAverage, self).__init__()
        self.window_size = window_size
        
        self.conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=window_size,
            padding=window_size // 2,  # Use padding to maintain same output size
            bias=False,
            groups=num_channels
        )
        
        # Set the weights to uniform values for average
        avg_filter = torch.ones((num_channels, 1, window_size)) / window_size
        self.conv.weight.data.copy_(avg_filter)
        
        # Freeze the weights (disable gradients for these parameters)
        for param in self.conv.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.conv(x)
        return x


class SlidingWindowAverageAndSubtract(nn.Module):
    def __init__(self, window_size: int, num_channels: int):
        super(SlidingWindowAverageAndSubtract, self).__init__()
        self.averager = SlidingWindowAverage(window_size, num_channels)

    def forward(self, x, mask):
        sliding_average = self.averager(x)
        if mask is not None:
            sliding_average.masked_fill_(~mask.unsqueeze(1), 0)
        subtract = x - sliding_average
        return subtract, sliding_average


if __name__ == "__main__":
    # Example usage with a random sequence
    window_size = 3  # Define the sliding window size
    num_channels = 2  # Number of input/output channels
    data = torch.randn(1, num_channels, 5)  # A random sequence of length 10
    
    # Initialize the sliding-window average module
    sliding_window_avg = SlidingWindowAverage(window_size, num_channels)
    
    # Compute the average
    result = sliding_window_avg(data)
    
    print("Input sequence:", data, sep="\n")
    print("Sliding-window average:", result, sep="\n")

