#!/bin/bash

SESSION_NAME="promptgenerate0"

tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME 'export SLURMD_NODENAME="root"' C-m
tmux send-keys -t $SESSION_NAME 'conda activate prompt' C-m
tmux send-keys -t $SESSION_NAME 'export CUDA_VISIBLE_DEVICES=1' C-m
tmux send-keys -t $SESSION_NAME 'chmod +x ./prompttrain0.sh' C-m
tmux send-keys -t $SESSION_NAME './skill/generateprobe.sh' C-m
tmux attach-session -t $SESSION_NAME
