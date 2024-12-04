import numpy as np
from torch.utils.data import DataLoader

def get_loaders(datapath, batch_size, dataset_, start_index, world_size):
    
    def _norm(data, max, min):
        data = -1. + (data - min)*2. / (max - min)
        return data  
   
    data = []
    for path in datapath:
        data.append(np.load(path))
    
    data = np.concatenate(data, axis=1)
    interval_length = data.shape[0]//world_size
    
    data = _norm(data, data.max(axis=(0,2,3),keepdims=True), data.min(axis=(0,2,3),keepdims=True))
    print(f"For rank {start_index}, the interval used is from {start_index*interval_length} : {(start_index+1)*interval_length}")
    
    train_dataloader = DataLoader(dataset_(data[start_index*interval_length : (start_index+1)*interval_length]), batch_size=batch_size, shuffle=True)
    
    return train_dataloader

def dl_iter(dl):
    while True:
        yield from dl 