# timeout 5400 python -W ignore PyG/gnn_train.py --dataset flickr --model sage --sampler neighbor --record --batch_num 0 --cpu_process 4 --n_sampler 1 --n_trainer 7
# timeout 5400 python -W ignore PyG/gnn_train.py --dataset reddit --model sage --sampler neighbor --record --batch_num 0 --cpu_process 2 --n_sampler 1 --n_trainer 8
# timeout 5400 python -W ignore PyG/gnn_train.py --dataset reddit --model sage --sampler neighbor --record --batch_num 0 --cpu_process 2 --n_sampler 1 --n_trainer 12
# timeout 5400 python -W ignore PyG/gnn_train.py --dataset reddit --model sage --sampler neighbor --record --batch_num 0 --cpu_process 2 --n_sampler 1 --n_trainer 15

python -W ignore PyG/sub_run.py --dataset flickr --record --batch_num 0 --a 2 --b 1 --c 6
python -W ignore PyG/sub_run.py --dataset reddit --record --batch_num 0 --a 2 --b 1 --c 8
python -W ignore PyG/sub_run.py --dataset reddit --record --batch_num 0 --a 2 --b 1 --c 12
python -W ignore PyG/sub_run.py --dataset reddit --record --batch_num 0 --a 2 --b 1 --c 15

python -W ignore PyG/gnn_train.py --dataset flickr --model sage --sampler neighbor --record --batch_num 0 --cpu_process 2 --n_sampler 1 --n_trainer 6