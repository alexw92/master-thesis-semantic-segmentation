REM run in G:\Tensorflow_Models\research
REM for finetuning batch_norm use bigger batchsize of at least 12

cd C:\Uni\Masterstudium\ma-werthmann\code\Segmentation_Models\deeplab
set PYTHONPATH=%PYTHONPATH%;%cd%\slim
cd ..

python deeplab\train.py ^
    --logtostderr ^
    --training_number_of_steps=90000 ^
    --train_split="train" ^
    --model_variant="xception_65" ^
    --atrous_rates=6 ^
    --atrous_rates=12 ^
    --atrous_rates=18 ^
    --output_stride=16 ^
    --decoder_output_stride=4 ^
    --train_crop_size=600 ^
    --train_crop_size=600 ^
    --train_batch_size=1 ^
    --initialize_last_layer=True ^
    --last_layers_contain_logits_only=True ^
    --fine_tune_batch_norm=False ^
    --base_learning_rate=0.0001 ^
    --dataset="world_tiny2k" ^
    --train_logdir=G:\ConvNet_Models\deeplab_osmtest ^
    --tf_initial_checkpoint=G:\ConvNet_Models\deeplabv3+_pretrained\model\model.ckpt-30358 ^
    --dataset_dir="G:\Tensorflow_Models\research\deeplab\datasets\deeplab_tiny_world2k\tfrecord"