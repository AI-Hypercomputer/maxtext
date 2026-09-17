import torch

grid_h = torch.arange(2) # [0, 1]
grid_w = torch.arange(2) # [0, 1]

grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
print("grid[0] (w/x):")
print(grid[0])
print("grid[1] (h/y):")
print(grid[1])
