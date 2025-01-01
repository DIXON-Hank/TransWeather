timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# exp_name="${exp_name}_${timestamp}"
exp_name="new_model_raindrop_0.001_T200"

python train.py -exp_name "${exp_name}" -train_batch_size 18 -learning_rate 0.001 -num_epochs 200 -seed 39