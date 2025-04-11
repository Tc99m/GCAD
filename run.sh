# SMD
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --data ./dataset/smd --name smd --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001

# SWaT
python main.py --seq_len 5 --pred_len 1 --pd_beta 0.5 --sample_p 0.2 --sparse_th 0.008 --test_stride 5 --data ./dataset/swat --name swat --n_block 6 --ff_dim 2048 --dropout 0 --learning_rate 0.0001

# SMAP
python main.py --seq_len 70 --pred_len 1 --pd_beta 1 --sample_p 0.2 --sparse_th 0.008 --test_stride 1 --data ./dataset/smap --name smap --n_block 6 --ff_dim 1024 --dropout 0 --learning_rate 0.0001

# PSM
python main.py --seq_len 30 --pred_len 5 --pd_beta 0.5 --sample_p 0.1 --sparse_th 0.005 --test_stride 10 --data ./dataset/psm --name psm --n_block 2 --ff_dim 128 --dropout 0 --learning_rate 0.0001

# PSM
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.002 --test_stride 1 --data ./dataset/msl --name msl --n_block 5 --ff_dim 1024 --dropout 0 --learning_rate 0.0001