python regression_test.py --dataset mag240M --cpu_process 1 --sampler neighbor --model sage --hidden 128 --neighbor 0 --layer 3
python regression_test.py --dataset mag240M --cpu_process 1 --sampler neighbor --model sage --hidden 128 --neighbor 1 --layer 2
python regression_test.py --dataset mag240M --cpu_process 2 --sampler neighbor --model sage --hidden 128 --neighbor 0 --layer 3
python regression_test.py --dataset mag240M --cpu_process 2 --sampler neighbor --model sage --hidden 128 --neighbor 1 --layer 2
python regression_test.py --dataset mag240M --cpu_process 4 --sampler neighbor --model sage --hidden 128 --neighbor 0 --layer 3
python regression_test.py --dataset mag240M --cpu_process 4 --sampler neighbor --model sage --hidden 128 --neighbor 1 --layer 2
python regression_test.py --dataset mag240M --cpu_process 8 --sampler neighbor --model sage --hidden 128 --neighbor 0 --layer 3
python regression_test.py --dataset mag240M --cpu_process 8 --sampler neighbor --model sage --hidden 128 --neighbor 1 --layer 2