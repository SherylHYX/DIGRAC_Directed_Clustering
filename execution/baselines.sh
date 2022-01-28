cd ../src/
../../parallel -j10 --resume-failed --results ../Output/baselines_syn_1000_1 --joblog ../joblog/baselines_syn_1000_1_joblog   python ./train.py --no-cuda  --N 1000 --size_ratio 1 --p 0.02 --F_style 'star' --K 5  --ambient 0  --eta {1} --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45


../../parallel -j4 --resume-failed --results ../Output/baselines_syn_1000_2 --joblog ../joblog/baselines_syn_1000_2_joblog  python ./train.py --no-cuda  --N 1000 --size_ratio 1.5 --p 0.1 --F_style 'complete' --K 5  --ambient 1  --eta {1} --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45

../../parallel -j4 --resume-failed --results ../Output/baselines_syn_1000_3 --joblog ../joblog/baselines_syn_1000_3_joblog  python ./train.py --no-cuda  --N 1000 --size_ratio 1.5 --p 0.02 --F_style 'cyclic' --K 3  --ambient 0  --eta {1} --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35
../../parallel -j3 --resume-failed --results ../Output/baselines_syn_1000_4 --joblog ../joblog/baselines_syn_1000_4_joblog  python ./train.py --no-cuda  --N 1000 --size_ratio 1 --p 0.02 --F_style 'path' --K 5  --ambient 0  --eta {1} --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden  ::: 0 0.05 0.1 0.15 0.2 0.25

../../parallel -j5 --resume-failed --results ../Output/baselines_syn_1000_5 --joblog ../joblog/baselines_syn_1000_5_joblog  python ./train.py --no-cuda  --N 1000 --size_ratio 1 --p 0.1 --F_style 'complete' --K 3  --ambient 0  --eta {1} --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45
../../parallel -j10 --resume-failed --results ../Output/baselines_syn_1000_6 --joblog ../joblog/baselines_syn_1000_6_joblog   python ./train.py --no-cuda  --N 1000 --size_ratio 1  --p 0.02 --F_style 'complete' --K 10  --ambient 1  --eta {1} --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 

../../parallel -j2 --resume-failed --results ../Output/baselines_syn_5000 --joblog ../joblog/baselines_syn_5000_joblog  python ./train.py --no-cuda  --N 5000 --size_ratio 1.5 --p 0.01 --F_style 'cyclic' --K 5  --ambient 1  --eta {1}  --tau 0.5 --hop 2 --hidden 32 --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45

../../parallel -j1 --resume-failed --results ../Output/baselines_syn_30000 --joblog ../joblog/baselines_syn_30000_joblog  python ./train.py --no-cuda  --N 30000 --size_ratio 1 --p 0.001 --F_style 'cyclic' --K 5  --ambient 0  --eta {1}  --tau 0.5 --hop 2 --hidden 32 --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3

../../parallel -j1 --resume-failed --results ../Output/baselines_real_data --joblog ../joblog/baselines_real_data_joblog python ./train.py --no-cuda  --dataset {1} --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden -SP -All  -a ::: wikitalk blog telegram migration 

../../parallel -j10 --resume-failed --results ../Output/baselines_lead_lag --joblog ../joblog/baselines_lead_lag_joblog  python ./train.py --no-cuda  --dataset lead_lag --all_methods Bi_sym DD_sym DISG_LR Herm Herm_rw InfoMap Louvain Leiden --year {1}  -SP -All ::: {0..18} 

