### Training

```shell script
python train.py --size 256 --data {dataset_name} --epoch 10 --batch_size 1 --lr 0.00008 --device {gpu_ids} --dst {experiment_name} --vit_root ./pretrained_detectors/{vit_detector_folder} --dif_root ./pretrained_detectors/{dif_detector_folder}
```
```size```: Referes to the **image size**. It is fixed at **256**.<br>
```epoch```: The number of training iterations. You can specify the desired number.<br>
```batch_size, lr```: The current values are to be kept fixed.<br>
```dst```: Specifies the folder where the results will be saved. If you enter a name, it will be automatically created inside the **results** folder. An error will occur if the folder name already exists within the **results** directory.<br>
```vit_root, dif_root```: The **pre-trained models** for the ViT and DIF detectors that PGD will attack. If the ```pretrained_detectors``` folder is missing, click the Drive link below to download it, and the move the downloaded ```pretrained_detectors``` folder into the ```STIG-PGD``` directory. <br>
[Demo file](https://drive.google.com/drive/folders/1q6-aDSPMqo_txCLIuriYCSQHHrsGAvaW?usp=sharing) <br>
Enter the command. You can change the GPU device by modifying the option ```--device```.<br>

The sampled results are visualized in the ```results/{experiment_name}/sample/``` during the training. The samples are saved every 100 steps.<br> 

After training, the results image and magnitude spectrum will be saved at each folder in ```results/{experiment_name}/eval/```.<br>

Inside the ```eval``` folder, you will find three subfolders: ```clean```, ```denoised```, and ```noise```. The ```clean``` folder contains the original real images, the ```noise``` folder contains the original AI-generated images, and the ```denoised``` folder contains the refined AI images.
### Inference
<p align="center">
    <img src="https://github.com/user-attachments/assets/716ba711-a323-400c-98e8-25f79fd27386" alt="Inference" width="80%" />
</p>

When you run the ```gradio_app.py``` file, the following interface appears. You can perform inference or evaluate the detector through this interface.<br> 

```모델 체크포인트 경로```: This is the checkpoint path for the trained model. It is a ```.pt``` format file located in ```results/{experiment_name}```. Please enter the absolute path of the file. e.g. ```E:\STIG-PGD\STIG-PGD\results\human\parameters_0_epoch.pt```<br>
```GPU 디바이스```: This is the GPU device number. It is usually 0.<br>
```이미지 크기```: The image size is fixed at 256.<br>
```입력 데이터셋 폴더 경로```: This is the absolute path to the folder containing the fake images to be used for inference. e.g. ```E:\STIG-PGD\STIG-PGD\datasets\human_inference\fake``` You need a fake dataset for inference that was not used in training.<br>
```추론 결과 저장 폴더 경로```: This is the absolute path to the folder where the inference result images will be saved. The inference result images are located in the ```denoised``` folder within the inference result folder.<br>

The inference results are saved to the location you designated as the inference results folder.<br>

Inside the inference results folder, you will find subfolder: ```denoised```. The ```denoised``` folder contains the refined AI images after inferencing.

### Detection(Evaluation)

We have prepared two detectors: the ViT and DIF detectors. <br>
You cna train two detectors or measure their performance using an existing trained model. <br>

#### Training Detectors

**ViT**
```shell script
python train_vit.py --is_train True --classifier vit --lr 0.0002 --size 256 --device {gpu_ids} --class_epoch {epochs} --class_batch_size 32 --dst {experiment_name}
```

```is_train, lr, size, class_batch_size```: Current value is fixed. <br>
```classifier```: Detector type is fixed to ViT. <br>
```device```: This is the GPU device number. It is usually 0. <br>
```class_epoch```: This is the number of training epochs. <br>
```dst```: This is the folder where the trained model results will be saved. Specify the target folder within the ```results``` folder that will be used for training. <br> 
Additionally, ViT uses the clean(real image) and noise(fake image) folders from the ```eval``` folder, which is located inside the target folder within the ```results``` directory, for training. The resulting model file is saved within the newly created ```vit_classifier``` folder inside the target folder.<br>

**DIF**
```shell script
python train_dif.py datasets/{dataset_name} checks/{checkpoint_name} --e {epochs} --f 1
```

```datasets```: This is the path to the dataset for training. Specify the folder name within the ```datasets``` folder to be used for training. <br>
```checks```: This is the path to save the training checkpoint. If the ```checks``` folder does not exist, please create it manually. The code will then save the checkpoint in a subfolder with the specified checkpoint folder name. <br>
```e```: This is the number of training epochs. <br>
```f```: This is the checkpoint saving frequency. Setting it to 1 will save a checkpoint every epoch.

#### Evaluation

After training, you can run gradio_app.py and perform evaluation in the detect tab.

<p align="center">
    <img src="https://github.com/user-attachments/assets/9569fcf9-8738-4b0a-96bc-fe83cb5131be" alt="Inference" width="80%" />
</p>

```감지기 모델 선택```: You can select the model to be evaluated. <br>
```감지기 모델 폴더 경로```: This is the absolute path where the detector model's checkpoint is saved. <br>
```Real 이미지 데이터셋 폴더 경로```: This is the absolute path to the real image dataset that the detector model will use for detection. <br>
```Fake 이미지 데이터셋 폴더 경로```: This is the absolute path to the fake image dataset that the detector model will use for detection. <br>

Once the evaluation is complete, you can review the sample results. You can check how the detector classified the real and fake images. The overall performance is displayed through the F1 score and Accuracy. <br>


