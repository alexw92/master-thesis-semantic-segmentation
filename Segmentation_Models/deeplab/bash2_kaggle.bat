REM run in G:\Tensorflow_Models\research
REM for finetuning batch_norm use bigger batchsize of at least 12

cd C:\Uni\Masterstudium\ma-werthmann\code\Segmentation_Models\deeplab
set PYTHONPATH=%PYTHONPATH%;%cd%\slim
cd ..

python deeplab\train.py ^
    --logtostderr ^
    --training_number_of_steps=90000 ^
    --train_split="train" ^
    --model_variant="mobilenet_v2" ^
    --output_stride=16 ^
    --decoder_output_stride=4 ^
    --train_crop_size=600 ^
    --train_crop_size=600 ^
    --train_batch_size=1 ^
    --initialize_last_layer=False ^
    --last_layers_contain_logits_only=True ^
    --fine_tune_batch_norm=False ^
    --base_learning_rate=0.0001 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
	--classweights=1.0 ^
    --dataset="kaggle_dstl" ^
    --train_logdir=G:\ConvNet_Models\deeplab_kaggletest_3 ^
    --tf_initial_checkpoint="G:\Pretrained weights\deeplab\deeplabv3_mnv2_cityscapes_train\model.ckpt" ^
    --dataset_dir="G:\Datasets\deeplab_DSTL_Challenge_Dataset\tfrecord"