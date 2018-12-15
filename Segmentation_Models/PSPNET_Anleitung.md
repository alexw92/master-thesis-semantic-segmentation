# PSPNET

[Quelle-Github](https://github.com/hellochick/PSPNet-tensorflow)

```
python inference.py --img-path=./input/osmtest.png --dataset osm --checkpoints=./model/pspnet101
```
Für Inferenz von einem Bild.


--[Queckpoint-file Github](https://github.com/hellochick/PSPNet-tensorflow/issues/16)

- loaded pretrained vals von google drive (https://drive.google.com/drive/folders/0B9CKOTmy0DyaV09LajlTa0Z2WFU)
- changed checkpoint.txt to checkpoint!

Beispiel für sbatch:

```
sbatch -p dmir --gres=gpu:1 run2.sh -e /scratch/tensorflow/venv_base 
-s ./tf_segmodels/PSPNet-tensorflow/inference.py 
-r master/code/requirements.txt 
-p tf_segmodels/PSPNet-tensorflow/ 
-a "--img-path /autofs/stud/werthmann/tf_segmodels/PSPNet-tensorflow/input/osmtest.png --dataset osm --checkpoints ./model/pspnet101"
```


## Training Implementierung

Für Training auf osm daten muss die klassenanzahl geändert werden!
Dazu darf beim Checkpoint file der letzte layer nicht geladen werden:

```
restore_var = [v for v in tf.global_variables() if 'conv6' not in v.name]
```

Dazu einfach das ```--not-restore-last``` flag benutzen.
Das Training mit den default-Werten mit dem Osm-Daten geht wie folgt:

```
python osm_train.py --not-restore-last
```

Man muss nur dafür sorgen dass in list ein entsprechendes file liegt, das die Trainingsdatennamen
beinhaltet.

Achtung: Einige Argumente sind noch buggy und da die Pfadangabe bei sbatch im run2.sh-Script etwas konfus
ist, lieber absolute Pfade verwenden als Argument oder keine Argumente übergeben und Pfade hardcoden.

Beispielrun für sbatch:

```
sbatch -p dmir --gres=gpu:1 run2.sh -e /scratch/tensorflow/venv_base 
-s ./master/code/Segmentation_Models/PSPNet/osm_train.py 
-r master/code/requirements.txt 
-p ./master/code/Segmentation_Models/PSPNet 
-a --not-restore-last

```
In diesem Fall darf kein " benutzt werden bei der Übergabe des Arguments bei *-a*, da dieses sonst mit an das Script übergeben wird
und der Parameter nicht erkannt wird.

* Benutzt 'poly' learning rate policy (siehe caffe)
siehe: https://stackoverflow.com/a/33711600
siehe: /caffe-master/src/caffe/proto/caffe.proto

## Evaluation

```
python evaluation.py --dataset osm

```

Evaluiert in der aktuellen Version die Prediction eines einzelnes Images mit einem Labelfile.
**TODO:** Dynamische Evaluierung eines ganzen OSM-Datensatzes
Metrik Mean-Intersection-Over-Union soll auch für andere Models evaluiert werden!
**TODO:** Dafür am besten [Diesen Code](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py)
vom Cityscapes Datensatz verwenden und für den OSM-Datensatz anpassen

## Trainingsverlauf
- **Start Slurm Run: 18.05.2018 morgens**
- trainiert mit 1600 (600x600)Images über 10000 Iterations (<3h mit 1080Ti von LS6)
- [SLURM_OUT_LOG](../train_logs_slurm/slurm_psp_10000_its_first_test.out)
- > pretrained mit cityscaped, Gewichte geladen wie oben
- > leider sehr schlechte Ergebnisse, benutzt: poly lr policy mit base_lr: 1e-3 und power=0.9
- > Ergebnisse loss schwankt im mittleren-oberen 2.000 - Bereich
- TODO: Ging er überhaupt runter später? Threshold? PLOTTEN!!Dann schauen ob LR Policy geändert werden sollte

____
- Neuer Trainingsrun gestartet (**Start Slurm Run: 18.05.2018**) mit 1600 (600x600)Images über 100000 Iterations (~35h mit 1080Ti von LS6)

Ist leider nicht komplett durchgelaufen. Letzter gespeicherter Checkpoint bei 77400.
**Wie verhindert man das? Wieso kommt überhaupt OOM?**

- [SLURM_OUT_LOG](../train_logs_slurm/slurm_psp_100000_sec_test.out)

```
step 77394 /t loss = 2.040, (1.130 sec/step)
step 77395 /t loss = 2.332, (1.092 sec/step)
step 77396 /t loss = 2.172, (1.098 sec/step)
step 77397 /t loss = 2.116, (1.138 sec/step)
step 77398 /t loss = 2.087, (1.109 sec/step)
step 77399 /t loss = 2.272, (1.130 sec/step)
The checkpoint has been created.
```

**Zur Frage:** Ging er überhaupt runter später? Threshold? PLOTTEN!!

**Antwort:** Er ging runter, wenn auch sehr langsam.
Die LR-Policy, die verwendet wurde sieht man hier
![LR-Policy](./PSPNet/plots_imgs/LR_Poly_Base_0_001_mom_0.9.png)

Hier ist die Learning-Curve vom ersten run mit 10000 iterations (~12.5 epochs, wegen batchsize=2)
![LR-Curve-Firstrun](./PSPNet/plots_imgs/LC_pspnet_firstrun_10000.png)

Hier der Plot vom Training mit 100000 images (wieder von scratch).
Parameter waren bei beiden runs gleich:

```
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
POWER = 0.9 #für lr-policy 'poly' ->siehe caffe
WEIGHT_DECAY = 0.0001
```

Zu Beachten ist, dass die Anzahl der images sehr klein war (1600) und kein Pretraining
und kein Class-Weighting vorgenommen wurde
![LR-Curve-Secrun](./PSPNet/plots_imgs/LC_pspnet_secrun_100000.png)


