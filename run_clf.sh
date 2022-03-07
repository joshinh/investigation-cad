GPUDEV=$1
SEED=$2
DATASET=$3
EXPMODEL=$4
DATAFORMAT=$5
INPREFIX=$6
TRAIN=$7
EVAL=$8
SAMPLENEGS=$9
TODROP="${10}"
MODELPATH="${11}"
CONTRAST="${12}"

MODELTYPE=roberta
MODELNAME=roberta-base
#MODELTYPE=bert
#MODELNAME=bert-base-uncased


if [ "$DATAFORMAT" == "instance" ] || [ "$DATAFORMAT" == "Explanation_1" ] || [ "$DATAFORMAT" == "hyp_only" ]
then
    SEQLEN=100
    TBSZ=32
    EBSZ=32
    INPREFIX=""
    INSUFFIX=""
    #if [ "$DATASET" != 'lit' ]
    #then
    #    INSUFFIX="_data"
    #fi
elif [ "$DATAFORMAT" == "all_explanation" ]
then
    SEQLEN=100
    TBSZ=32
    EBSZ=32
    INSUFFIX="_merged_all"
elif [ "$DATAFORMAT" == "independent" ] || [ "$DATAFORMAT" == "aggregate" ]
then
    SEQLEN=50
    TBSZ=32
    EBSZ=32
    INSUFFIX="_merged"
elif [ "$DATAFORMAT" == "append" ]
then
    SEQLEN=100
    TBSZ=32
    EBSZ=32
    INSUFFIX="_merged"
elif [ "$DATAFORMAT" == "instance_independent" ] || [ "$DATAFORMAT" == "instance_aggregate" ]
then
    SEQLEN=100
    TBSZ=16
    EBSZ=16
    INSUFFIX="_merged"
elif [ "$DATAFORMAT" == "instance_append" ]
then
    SEQLEN=200
    TBSZ=16
    EBSZ=16
    INSUFFIX="_merged"
fi

NEPOCHS=10

TRAINFILE=./data/dataset_"$DATASET"/"$INPREFIX""$TRAIN""$INSUFFIX".csv
if [ "$TRAIN" == "_" ]
then
    TRAINCMD=""
else
    TRAINCMD="--do_train"
fi
EVALFILE=./data/dataset_"$DATASET"/"$INPREFIX""$EVAL""$INSUFFIX".csv

if [ "$SAMPLENEGS" == "sample" ]
then
    SAMPLECMD="--sample_negs"
    SAMPLESTR="_negs"
else
    SAMPLECMD=""
    SAMPLESTR=""
fi

if [ "$CONTRAST" == "contrast" ] && [ "$TRAIN" == "_" ]
then
    CONTRASTCMD="--contrast"
    CONTRASTSTR="contrast"
elif [ "$CONTRAST" == "contrast" ]
then
    CONTRASTCMD="--contrast"
    CONTRASTSTR="contrast"
    TBSZ=6
    TRAINFILE=./data/dataset_"$DATASET"/all/"$INPREFIX""$TRAIN""$INSUFFIX"_reordered.csv
else
    CONTRASTCMD=""
    CONTRASTSTR=""
fi

if [ "$TODROP" == "_" ]
then
    TODROPCMD=""
else
    TODROPCMD="--to_drop "$TODROP""
fi

if [ "$MODELPATH" == "_" ]
then
 OUTPUTDIR="./saved_clf/seed"$SEED"_"$DATASET"_"$TRAIN"_"$TBSZ"_"$SEQLEN"_"$NEPOCHS""
else
    OUTPUTDIR="$MODELPATH"
fi


cmd="CUDA_VISIBLE_DEVICES=$GPUDEV python3 run_nli.py "$SAMPLECMD" "$TODROPCMD" "$CONTRASTCMD" \
    --cache_dir ../nile_release/cache \
    --seed "$SEED" \
    --model_type "$MODELTYPE"  \
    --model_name_or_path "$MODELNAME" \
    --exp_model "$EXPMODEL" \
    --data_format "$DATAFORMAT" \
    "$TRAINCMD" --save_steps 1523000000000000 \
    --do_eval \
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
