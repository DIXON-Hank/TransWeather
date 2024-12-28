timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

exp_name="debug_${timestamp}"
# exp_name="debug"

python train.py -exp_name "${exp_name}" -train_batch_size 24 -learning_rate 0.0002 -num_epochs 50