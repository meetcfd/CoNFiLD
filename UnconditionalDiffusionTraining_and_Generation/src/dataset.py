from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
 
class VF(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self._preprocess(data,'data')
        
    def _preprocess(self, data, name):
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index]

DATASETS = {"VF": VF}

if __name__ == "__main__":
    pass