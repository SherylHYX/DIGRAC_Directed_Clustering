cd ../src/
python train.py -D --regenerate_data --no_cuda
python train.py -D --normalizations vol_min vol_sum vol_max plain --thresholds sort std naive
python train.py -D -All --dataset blog --all_methods MagNet DGCN DiGCN DIGRAC DiGCL
python train.py -D  --load_only -SP