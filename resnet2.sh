#!/bin/bash
#PBS -N simclrv2_resnet50_pretrain
#PBS -j oe
#PBS -m abe
#PBS -M hieulenho8@gmail.com
#PBS -l select=4:ncpus=4:ngpus=1:mpiprocs=1:mem=32G
#PBS -q para_gpu

cd "$PBS_O_WORKDIR" || exit 1

mkdir -p checkpoints logs

LOG_FILE="logs/${PBS_JOBNAME}_${PBS_JOBID}.log"
exec > "$LOG_FILE" 2>&1

set -ex

PYTHON=/opt/apps/python/3.8.10/bin/python3
export PATH=/opt/apps/python/3.8.10/bin:$PATH
export OMP_NUM_THREADS=4

# One MPI rank per unique PBS node, each rank controls one GPU.
UNIQUE_NODEFILE="${PBS_O_WORKDIR}/pbs_unique_nodes_${PBS_JOBID}.txt"
sort -u "$PBS_NODEFILE" > "$UNIQUE_NODEFILE"
export MASTER_ADDR="$(head -n 1 "$UNIQUE_NODEFILE")"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="$(wc -l < "$UNIQUE_NODEFILE" | tr -d ' ')"

echo "===== JOB INFO ====="
echo "Job ID: $PBS_JOBID"
echo "Job name: $PBS_JOBNAME"
echo "Workdir: $PBS_O_WORKDIR"
echo "Master addr: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Start time: $(date)"
echo "Current dir: $(pwd)"
echo "PBS_NODEFILE=$PBS_NODEFILE"
echo "UNIQUE_NODEFILE=$UNIQUE_NODEFILE"
cat "$UNIQUE_NODEFILE"

echo "===== FILE CHECK ====="
ls -lah
test -f train_resnet.py
test -f dataset.py
test -f ssl_simclr.py
test -d resnet18

echo "===== PYTHON ====="
echo "PYTHON=$PYTHON"
"$PYTHON" --version
"$PYTHON" -c "import sys; print(sys.executable)"

echo "===== TORCH/CUDA ====="
"$PYTHON" -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count()); print('gpu0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"

echo "===== GPU ====="
nvidia-smi || true

echo "===== START TRAIN ====="

mpirun -np "$WORLD_SIZE" -ppn 1 -f "$UNIQUE_NODEFILE" \
  "$PYTHON" train_resnet.py \
    --mode pretrain \
    --distributed \
    --arch resnet50 \
    --dataset stl10 \
    --data-root ./data \
    --download \
    --image-size 96 \
    --epochs 500 \
    --batch-size 256 \
    --optimizer lars \
    --lr 0.05 \
    --warmup-epochs 15 \
    --weight-decay 1e-4 \
    --proj-layers 3 \
    --proj-dim 128 \
    --temperature 0.2 \
    --num-workers 4 \
    --out-dir ./checkpoints \
    --save-every 20 \
    --log-every 20

echo "===== END TRAIN ====="
echo "End time: $(date)"
