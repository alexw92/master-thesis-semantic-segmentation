REM run in G:\Tensorflow_Models\research
REM for finetuning batch_norm use bigger batchsize of at least 12

cd G:\Tensorflow_Models\research
set PYTHONPATH=%PYTHONPATH%;%cd%\slim
cd ..

python train.py ^
    --logtostderr ^
    --training_number_of_steps=90000 ^
    --train_split="train" ^
    --model_variant="xception_65" ^
    --atrous_rates=6 ^
    --atrous_rates=12 ^
    --atrous_rates=18 ^
    --output_stride=16 ^
    --decoder_output_stride=4 ^
    --train_crop_size=769 ^
    --train_crop_size=769 ^
    --train_batch_size=1 ^
    --fine_tune_batch_norm=False ^
    --dataset="cityscapes" ^
    --train_logdir=G:\ConvNet_Models\deeplab_testmodel ^
    --tf_initial_checkpoint=G:\ConvNet_Models\deeplabv3+_pretrained\model\model.ckpt-30358 ^
    --dataset_dir="C:\Uni\Masterstudium\Master Datasets\Cityscapes\tfrecord"