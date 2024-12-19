#Imports
import os
import sys
import torch
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion
from src.train_util import TrainLoop
from src.dataset import DATASETS
from src.dataloader import get_loaders, dl_iter
from src.dist_util import setup_dist, dev
from src.logger import configure, log
from basicutility import ReadInput as ri

## Setup
torch.manual_seed(42)
np.random.seed(42)

inp = ri.basic_input(sys.argv[1])

setup_dist()
configure(dir=inp.log_path, format_strs=["stdout","log","tensorboard_new"])

## HyperParams (Change according to the case)
batch_size = inp.batch_size
image_size= inp.image_size
num_channels= inp.num_channels
num_res_blocks= inp.num_res_blocks
num_head_channels= inp.num_head_channels
attention_resolutions= inp.attention_resolutions
channel_mult = inp.channel_mult
use_scale_shift_norm = inp.use_film
use_new_attention_order = inp.use_new_attn

steps= inp.steps
noise_schedule= inp.noise_schedule

microbatch= inp.microbatch
lr = inp.lr
ema_rate= inp.ema_rate
log_interval= inp.log_interval
save_interval= inp.save_interval
lr_anneal_steps= inp.lr_anneal_steps

## Data Preprocessing
log("creating data loader...")
dl_train = get_loaders(datapath=inp.train_data_path, batch_size=inp.batch_size, dataset_=DATASETS[inp.dataset],
                       start_index=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

## Unet Model
log("creating model and diffusion...")

unet_model = create_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions,
                          channel_mult=channel_mult,
                          use_scale_shift_norm=use_scale_shift_norm,
                          use_new_attention_order=use_new_attention_order
                        )

unet_model.to(dev())

## Gaussian Diffusion
diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule
                                    )

## Training Loop
log("training...")

train_uncond_model = TrainLoop(
                                model=unet_model,
                                diffusion=diff_model,
                                train_data = dl_iter(dl_train),
                                batch_size= batch_size,
                                microbatch= microbatch,
                                lr = lr,
                                ema_rate=ema_rate,
                                log_interval=log_interval,
                                save_interval=save_interval,
                                lr_anneal_steps=lr_anneal_steps,
                                resume_checkpoint=inp.model_path if inp.model_path is not None else "")

train_uncond_model.run_loop()