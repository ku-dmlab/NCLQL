project_name="NCLQL"

envs=("HalfCheetah-v4")
# "Ant-v4" "Hopper-v4" "Humanoid-v4" "Walker2d-v4" "Swimmer-v4"
seeds=(100)
#  200 300 400 500

gpus=(0 1 2 3)
workers_per_gpu=4

jobs=()

for s in "${seeds[@]}"; do
  for env in "${envs[@]}"; do
    jobs+=("$env $s")
  done
done

mapfile -t jobs < <(printf '%s\n' "${jobs[@]}" | sort -u)
num_jobs=${#jobs[@]}
echo "deduped jobs: $num_jobs"

num_jobs=${#jobs[@]}
num_gpus=${#gpus[@]}

for ((gpu_idx=0; gpu_idx<num_gpus; gpu_idx++)); do
  for ((worker=0; worker<workers_per_gpu; worker++)); do
    (
      for ((job_idx=worker*num_gpus+gpu_idx; job_idx<num_jobs; job_idx+=num_gpus*workers_per_gpu)); do
        IFS=' ' read -r env s <<< "${jobs[$job_idx]}"
        echo "GPU ${gpus[$gpu_idx]} (worker $worker): $env $s"
        XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
        CUDA_VISIBLE_DEVICES=${gpus[$gpu_idx]} \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 \
        python scripts/train_mujoco.py --alg NCLQL --project_name="$project_name" --env "$env" --seed="$s"
        sleep 2
      done
    ) &
    sleep 2
  done
done

wait
echo "All jobs finished."