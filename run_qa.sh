GPUDEV=$1
SEED=$2
DATASET=$3
TRAIN=$4
EVAL=$5
MODELPATH="${6}"

MODELTYPE=roberta
MODELNAME=roberta-base
SEQLEN=350
TBSZ=12
EBSZ=12
NEPOCHS=10

TRAINFILE=./data/dataset_"$DATASET"/heldout/"$TRAIN".csv
EVALFILE=./data/dataset_"$DATASET"/perturbations/"$EVAL".csv
if [ "$TRAIN" == "_" ]
then
    TRAINCMD=""
else
    TRAINCMD="--do_train"
fi


if [ "$MODELPATH" == "_" ]
then
    OUTPUTDIR="./saved_clf/seed"$SEED"_"$DATASET"_"$TBSZ"_"$SEQLEN"_"$NEPOCHS""
else
    OUTPUTDIR="$MODELPATH"
fi


cmd="CUDA_VISIBLE_DEVICES=$GPUDEV python3 run_qa.py \
    --cache_dir ../nile_release/cache \
    --seed "$SEED" \
    --model_type "$MODELTYPE"  \
    --model_name_or_path "$MODELNAME" \
    "$TRAINCMD" --save_steps 1523000000000000 \
    --do_eval --eval_all_checkpoints \
    --do_lower_case \
    --train_file "$TRAINFILE"  --eval_file "$EVALFILE" \
    --max_seq_length "$SEQLEN" \
    --per_gpu_eval_batch_size="$EBSZ"   \
    --per_gpu_train_batch_size="$TBSZ"   \
    --learning_rate 2e-5 \
    --num_train_epochs "$NEPOCHS" \
    --logging_steps 100000 --evaluate_during_training \
    --output_dir "$OUTPUTDIR" \
    --save_every_epoch "
echo $cmd
eval $cmd

