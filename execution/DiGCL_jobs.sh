cd ../src/
../../parallel -j5 --resume-failed --results ../Output/DiGCL_syn_1000_1 --joblog ../joblog/DiGCL_syn_1000_1_joblog   CUDA_VISIBLE_DEVICES=3 python ./train.py   --N 1000 --size_ratio 1 --p 0.02 --F_style 'star' --K 5  --ambient 0  --eta {1} --all_methods DiGCL  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45

../../parallel -j4 --resume-failed --results ../Output/DiGCL_syn_1000_2 --joblog ../joblog/DiGCL_syn_1000_2_joblog  CUDA_VISIBLE_DEVICES=3 python ./train.py   --N 1000 --size_ratio 1.5 --p 0.1 --F_style 'complete' --K 5  --ambient 1  --eta {1} --all_methods DiGCL  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45

../../parallel -j4 --resume-failed --results ../Output/DiGCL_syn_1000_3 --joblog ../joblog/DiGCL_syn_1000_3_joblog  CUDA_VISIBLE_DEVICES=3 python ./train.py   --N 1000 --size_ratio 1.5 --p 0.02 --F_style 'cyclic' --K 3  --ambient 0  --eta {1} --all_methods DiGCL  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35
../../parallel -j3 --resume-failed --results ../Output/DiGCL_syn_1000_4 --joblog ../joblog/DiGCL_syn_1000_4_joblog  CUDA_VISIBLE_DEVICES=3 python ./train.py   --N 1000 --size_ratio 1 --p 0.02 --F_style 'path' --K 5  --ambient 0  --eta {1} --all_methods DiGCL  ::: 0 0.05 0.1 0.15 0.2 0.25

../../parallel -j4 --resume-failed --results ../Output/DiGCL_syn_1000_5 --joblog ../joblog/DiGCL_syn_1000_5_joblog  CUDA_VISIBLE_DEVICES=3 python ./train.py   --N 1000 --size_ratio 1 --p 0.1 --F_style 'complete' --K 3  --ambient 0  --eta {1} --all_methods DiGCL  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45
../../parallel -j5 --resume-failed --results ../Output/DiGCL_syn_1000_6 --joblog ../joblog/DiGCL_syn_1000_6_joblog   CUDA_VISIBLE_DEVICES=3 python ./train.py   --N 1000 --size_ratio 1  --p 0.02 --F_style 'complete' --K 10  --ambient 1  --eta {1} --all_methods DiGCL  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 
../../parallel -j2 --resume-failed --results ../Output/DiGCL_syn_5000 --joblog ../joblog/DiGCL_syn_5000_joblog  CUDA_VISIBLE_DEVICES=3 python ./train.py   --N 5000 --size_ratio 1.5 --p 0.01 --F_style 'cyclic' --K 5  --ambient 1  --eta {1}  --tau 0.5 --hop 2 --hidden 32 --all_methods DiGCL  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45


../../parallel -j1 --resume-failed --results ../Output/DiGCL_syn_30000 --joblog ../joblog/DiGCL_syn_30000_joblog  CUDA_VISIBLE_DEVICES=3 python ./train.py   --N 30000 --size_ratio 1 --p 0.001 --F_style 'cyclic' --K 5  --ambient 0  --eta {1}  --tau 0.5 --hop 2 --hidden 32 --all_methods DiGCL  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3
