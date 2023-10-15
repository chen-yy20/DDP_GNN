# Run experiments on PyG

1. Create the environment

   Run:

      ```bash
      conda env create -f environment.yml
      ```

      Then you will get the `GNN` environment.

      Run

      ```bash
      conda activate GNN
      ```

      to activate the environment. 

2. Create a Result folder

   ```
   mkdir PyG/Result
   ```

   ---
> Tip: You may need to input "yes" to download datasets first time running the program.

> Tip: Remember to modify the CPU configuration and dataset path in the program before running each program.

3. Scalability test 
   Run `gnn_train_scal.py` with different trainer core numbers for scalability test.
   Example: 

   ```
   python -W ignore gnn_train_scal.py --dataset ogbn-products --model sage --sampler neighbor --trainer 4
   ```

   The result would be recorded in `PyG/Result/Scal_<dataset>_<model>.txt`

4. Default experiment

   First time running the code may need to input `yes` to download the `ogbn` datasets.
   For default configuration, we set socket_core_num as trainer_core_num.
   Example (32 cores per CPU socket):

   ```
   python PyG/gnn_train_default.py --dataset ogbn-products --model sage --sampler neighbor --trainer 32
   ```

   The result would be recorded in `PyG/Result/Default_<dataset>_<model>.txt`

5. Bayesian Optimization
   Run `bo.py` for Bayesian Optimization.
   Example:

   ```
   python -W ignore PyG/bo.py --dataset flickr --model sage --sampler neighbor --batch_num 0
   ```
   `--batch_num 0` means measuring the training time for an entire epoch.
   You can set `--batch_num` when running BO on large dataset which take too much time to train. After finding the best configuration, run `gnn_train.py` to get the full epoch time. 

   The result would be saved in `PyG/Result/Bo_<dataset>_<model>.txt`.

   Single datapoint test:

   ```bash
   python PyG/gnn_train.py --dataset <dataset> --model <model> --sampler <sampler> --cpu_process <> --n_sampler <> --n_trainer <>  
   ```

6. Random search

   Basically the same as BO search. 
   Exampler:

   ```bash
   python -W ignore PyG/sim_annel.py --dataset ogbn-products --model sage --sampler neighbor --batch_num 0
   ```

   Result would be saved in `PyG/Result/Random_<dataset>_<model>.txt`

7. Grid search


   First, run

   ```bash
   python verify_gen.py
   ```
   to generate the bash file as `grid_search.sh`.  

   Then, run

   ```bash
   bash PyG/grid_search.sh
   ```

   Result would be saved in `PyG/Result/Grid_<dataset>_<model>.csv`

