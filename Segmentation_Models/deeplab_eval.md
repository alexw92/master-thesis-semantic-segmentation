Gestartet ca 2 Uhr 3.08.2018 mit 4(!) gtx1080ti 


**Applied slurm call:**
```
sbatch -p dmir --gres=gpu:4 run2.sh -e /scratch/tensorflow/venv_base\
 -s ./master/code/Segmentation_Models/deeplab/train.py\
 -r master/code/requirements.txt\
 -n True -i "deeplab train bn 12 batchsize 150k steps"\
 -p /autofs/stud/werthmann/master/code/Segmentation_Models:/autofs/stud/werthmann/master/code/Segmentation_Models/deeplab/slim \
 -a " --logtostderr \
--training_number_of_steps=150000 \
--train_split=train \
--model_variant=xception_65 \
--num_clones=4 \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--train_crop_size=609 \
--train_crop_size=609 \
--train_batch_size=12 \
--initialize_last_layer=True \
--last_layers_contain_logits_only=True \
--fine_tune_batch_norm=True \
--base_learning_rate=0.001 \
--dataset=de_top15 \
--train_logdir=./models/ES_deeplab_test_traindir \
--tf_initial_checkpoint=$HOME/master/code/Segmentation_Models/deeplab/models/ES_deeplab_test/model.ckpt-30358 \
--dataset_dir=/autofs/stud/werthmann/master/ANN_DATA/deeplab_de_top15_cropped/tfrecord"
```


- Pretrained cityscapes weights
- model_variant=xception_65 
- lr=0.001
- atrous_rates=6 
- atrous_rates=12 
- atrous_rates=18 
- output_stride=16 
- decoder_output_stride=4 
- batchsize=12 (minimum fuerr batchnorm)
- finetune batchnorm
- de_top15 train set
- **Wichtig**: Gewichte mit 100 skaliert : 
    **vorher (hier hat loss nur oszilliert):**   
    loss_weight = tf.constant([0.975644, 1.025603, 0.601745, 6.600600, 1.328684, 0.454776])
	**aktuell:**
    loss_weight = tf.constant([97.5644, 102.5603, 60.1745, 660.0600, 132.8684, 45.4776])
- **Geplant:** Bei Konvergenz das Model speichern und von dort aus mit andere Learning Rate/ Parametern spielen
	


miou: ~0.156    - ckpt-77000

miou:  0.194554 - ckpt-90110

miou:  0.25175  - ckpt-96339

miou:  0.18788  - ckpt-104120 schade, mal schauen ob es trotzdem noch weiter geht

miou:  0.25851	- ckpt-150000 gespeichert unter deeplab_run_0308_bn12b_topde15		


(Leider kann ich nur mit max 4 Karten trainieren :/)
Gewichte sind erst mal gleich

**Fortsetzung Run a) Slurm 61211** : 
Das ganze mit gleicher Config nochmal fuer 150k runs auf dmir laufen lassen

**Fortsetzung Run b) Slurm 61217** :
Das ganze mit **lr=0.0001** (vorher 0.001) fuer 150k runs auf paramsearch laufen lassen

**Ergebnis:**
Sowohl bei a als auch bei b oszilliert miou sehr stark
Die Ergebnisse sind bei a aber besser.
Run b abgebrochen bei step 230k und fortgesetzt mit batchsize=8, kein batchnorm, lr=0.0001, kleine weights (nicht mehr mal 100)




