import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """\
# DifFace Kaggle Training (T4x2) - 256x256 Resolution
This notebook is specifically configured to run on Kaggle's dual T4 GPU instances. It trains the DifFace IDDPM model on a dataset of 256x256 FFHQ images within a 2-hour bounds.

**Instructions:**
1. Ensure the Kaggle instance has Internet Enabled.
2. Attach your `FFHQ 256` dataset to the notebook using the Kaggle "Add Data" button.
3. Replace the `kaggle_dataset_path` variable in the data preparation cell with the actual path to your dataset.
"""

code_1 = """\
!git clone https://github.com/zsyOAOA/DifFace.git
%cd DifFace
!pip install timm kornia scikit-image einops gdown omegaconf loguru scipy
!pip install --upgrade torch torchvision
"""

text_2 = """\
## Data Configuration
Symlinking your attached FFHQ 256x256 dataset into the working directory so `main.py` can find it easily. 
**Change `kaggle_dataset_path` to match your actual input directory structure.**
"""

code_2 = """\
import os

# -------- ACTION REQUIRED --------
# Adjust this path based on what your attached Kaggle dataset is called.
# To find it, look in the data pane on the right side of the screen in Kaggle under /kaggle/input/
kaggle_dataset_path = "/kaggle/input/ffhq-256-images" 
# ---------------------------------

os.makedirs('./data/FFHQ256', exist_ok=True)
os.system(f"ln -s {kaggle_dataset_path}/* ./data/FFHQ256/")

print("Dataset symlinked to ./data/FFHQ256/")
"""

text_3 = """\
## YAML Configuration Override
Kaggle's T4 GPUs have 16GB of VRAM each. The original training script requires 8 GPUs and is unbounded. 
This block rewrites the configuration for a dual GPU setup by enabling mixed precision and limiting iterations to output a model checkpoint.
"""

code_3 = """\
import yaml
from omegaconf import OmegaConf

conf = OmegaConf.load('configs/training/diffusion_ffhq512.yaml')

# Modify resolution
conf.model.params.image_size = 256

# VRAM Optimizations for T4 GPUs
conf.train.use_fp16 = True
conf.model.params.use_fp16 = True

# Adjusting batch size for 2 GPUs
# The paper uses global batch 64, microbatch 8.
# For T4x2 at 256 resolution, we can likely fit a microbatch of 16 without OOM.
conf.train.microbatch = 16
conf.train.batch = [32, 4]  # [train_batch, val_batch]

# Point to attached dataset
conf.data.train.params.dir_path = './data/FFHQ256'

# Restrict run to ~2 hours. 
# 1 iteration takes approximately 0.5 - 1 seconds in DDP mode. 
# 5000 iterations should take ~1.5 hours, giving time to save checkpoints safely.
conf.train.iterations = 5000 
conf.train.save_freq = 1000
conf.train.val_freq = 1000

# Save out the new configured YAML
with open('configs/training/kaggle_diffusion_256.yaml', 'w') as f:
    OmegaConf.save(conf, f)

print("Kaggle Optimized YAML saved!")
"""

text_4 = """\
## Start Training Pipeline
Bootup PyTorch distributed training utilizing both T4 GPUs on this single node.
"""

code_4 = """\
!torchrun --standalone --nproc_per_node=2 --nnodes=1 main.py --cfg_path configs/training/kaggle_diffusion_256.yaml --save_dir ./kaggle_logs
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4)
]

with open('DifFace_Kaggle_Training.ipynb', 'w') as f:
    nbf.write(nb, f)
