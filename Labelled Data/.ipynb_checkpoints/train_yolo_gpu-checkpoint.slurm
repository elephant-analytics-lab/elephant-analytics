#!/bin/bash
#SBATCH --job-name=yolo_gpu          # job name
#SBATCH --ntasks=1                   # number of tasks
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=8gb                    # memory
#SBATCH --time=01:00:00              # max wall time (hh:mm:ss)
#SBATCH --gres=gpu:1                 # *** request 1 GPU ***
#SBATCH --output=yolo_gpu_%j.log     # log file (%j = job id)

# go to the folder that contains train_yolo.py
cd "/elephant/elephant-analytics/Labelled Data"

echo "Host: $(hostname)"
echo "Dir: $(pwd)"

# quick check: does this Python see CUDA?
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"

# run your YOLO training script
python3 train_yolo.py
