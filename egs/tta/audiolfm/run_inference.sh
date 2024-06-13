
######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

exp_config="$exp_dir/exp_config_lfm.json"


######## Run inference ###########
python "${work_dir}"/bins/tta/inference.py \
    --config $exp_config \
    --devices 3 \
    --checkpoint_file "checkpoint_path" \
    --num_steps 10 \
    --guidance_scale 3 \
    --infer \
    --text "Birds are chirping." 
