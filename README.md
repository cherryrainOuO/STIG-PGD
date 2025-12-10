# Attacking Fake-Image Detectors with STIG-PGD

<p align="center">
    <img src="https://github.com/user-attachments/assets/7a03e788-baa7-4ce7-9a09-a8286cd6b6a0" alt="Result 1" width="80%">
    <img src="https://github.com/user-attachments/assets/853d8507-81cc-4dca-ae74-c32cfd47e7fa" alt="Result 3" width="80%">
    <img src="https://github.com/user-attachments/assets/77edcbc0-b746-4c97-9017-db8786bc6590" alt="Result 5" width="80%">
    <img src="https://github.com/user-attachments/assets/57d7eea8-e8c5-4ce6-bdd3-80c8b07ba0c6" alt="Result 7" width="80%">
</p>

## Description
### 🚀 Project Overview
This repository contains the official PyTorch implementation of the STIG-PGD method, a novel adversarial image transformation technique designed to neutralize state-of-the-art fake image detectors.

Our method uniquely combines Spectral Transformation for refinement of Image Generation (STIG) and Projected Gradient Descent (PGD) to create Refined Fake Images that evade detection systems focusing on both spectral artifacts and visual inconsistencies.

### Core Contributions
**Hybrid Attack Superiority**: Achieves a significantly higher attack success rate against fake image detectors compared to single-technique approaches.

**Frequency-Selective Artifact Injection**: Addresses the limitations of traditional PGD by applying adversarial artifacts selectively to the Low-Frequency (LF) spectrum while using STIG to refine the High-Frequency (HF) spectrum.

**Novel Loss Function**: Introduces a hybrid PGD loss that integrates the STIG framework's Reconstruction Loss, ensuring the generated artifacts complement spectral refinement.

**Vulnerability Proof**: Demonstrates the fragility of current fake image detection models, guiding future research toward more robust defense mechanisms.

### STIG-PGD Framework
The framework is largely composed of the **STIG Framework**, the **PGD Framework**, and the **result merging step**.

<img width="1107" height="507" alt="image" src="https://github.com/user-attachments/assets/30c8ea5a-a2b0-48c9-9a33-222c5197cb13" />

**1. STIG Framework**

The STIG component refines the fake image's spectral quality. It uses Fast Wavelet Transform (FWT) for low-loss conversion and Patch-wise Contrastive Learning to align the High-Frequency (HF) spectrum with real images, preserving image structure.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1bc2eea2-e755-4527-9dfa-c9aa22bdc686" width="465" height="223" alt="image" />
</p>

**2. PGD Framework**

PGD generates adversarial perturbations. It updates the gradient toward the Real Class using a Hybrid Loss Function that combines the Detector's Classification Loss with STIG's Reconstruction Loss to produce the PGD Artifact Coefficient.

<p align="center">
<img width="185" height="191" alt="image" src="https://github.com/user-attachments/assets/6ba33806-c174-442f-ad53-4c8ca1042942" />
</p>

**3. Merging (The Core Innovation)**

This step strategically combines the STIG and PGD results to mitigate noise. It applies the PGD Artifact Coefficient only to the Low-Frequency (LF) region of the STIG spectrum, while maintaining the STIG-refined spectrum in the HF region. The combined result yields the final Refined Fake Image via Inverse Wavelet Transform (IWT).

<p align="center">
<img width="234" height="238" alt="image" src="https://github.com/user-attachments/assets/cf7d7738-a6ce-44a1-86e5-afc94455e72d" />
</p>

**4. Evasion Verification**

The final Refined Fake Image is confirmed to be effective by forcing the ViT, DIF Detector to misclassify it as Real.

<p align="center">
<img width="321" height="113" alt="image" src="https://github.com/user-attachments/assets/e8ef8ffd-2251-4780-897b-a43596edf372" />
</p>

## Performance Comparison
<table>
    <thead>
        <tr>
            <th align="center" rowspan="2">Test Images</th>
            <th align="center" colspan="2">ViT</th>
            <th align="center" colspan="2">DIF</th>
            <th align="center" colspan="2">Average</th>
            <th align="center" rowspan="2">Remark</th>
        </tr>
        <tr>
            <th align="center">Accuracy</th>
            <th align="center">F1-Score</th>
            <th align="center">Accuracy</th>
            <th align="center">F1-Score</th>
            <th align="center">Accuracy</th>
            <th align="center">F1-Score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Real Images (2400) <br> + Original AI Generated Images (2400)</td>
            <td align="center">93.5%</td>
            <td align="center">0.9351</td>
            <td align="center">99.6%</td>
            <td align="center">0.9965</td>
            <td align="center">96.57%</td>
            <td align="center">0.9658</td>
            <td align="left">Baseline accuracy against unaugmented fake images.</td>
        </tr>
        <tr>
            <td align="center">Real Images (2400) <br> + STIG Augmented AI Images (2400)</td>
            <td align="center">81.1%</td>
            <td align="center">0.8107</td>
            <td align="center">89.1%</td>
            <td align="center">0.8782</td>
            <td align="center">85.1%</td>
            <td align="center">0.8444</td>
            <td align="left">Spectral refinement offers moderate evasion.</td>
        </tr>
        <tr>
            <td align="center">Real Images (2400) <br> + PGD Augmented AI Images (2400)</td>
            <td align="center">74.9%</td>
            <td align="center">0.746</td>
            <td align="center">74.7%</td>
            <td align="center">0.6626</td>
            <td align="center">74.8%</td>
            <td align="center">0.7043</td>
            <td align="left">PGD artifacts significantly decrease detection performance.</td>
        </tr>
        <tr>
            <td align="center">Real Images (2400) <br> + STIG-PGD Augmented AI Images (2400)</td>
            <td align="center"><b>43.5%</b></td>
            <td align="center"><b>0.3033</b></td>
            <td align="center"><b>63.02%</b></td>
            <td align="center"><b>0.4147</b></td>
            <td align="center"><b>53.26%</b></td>
            <td align="center"><b>0.359</b></td>
            <td align="left"><b>Lowest F1-Score 👑</b></td>
        </tr>
    </tbody>
</table>

## Additional Results

<p align="center">
<img width="80%" alt="6_7500" src="https://github.com/user-attachments/assets/5926dfa5-e533-4e00-991e-81f00ffb452c" />
<img width="80%" alt="6_6400" src="https://github.com/user-attachments/assets/c439f107-bb24-4e02-a23c-4ba00b23974b" />
<img width="80%" alt="6_5400" src="https://github.com/user-attachments/assets/eafa4d0a-5dfe-4ff6-b0b0-1e1229b65357" />
<img width="80%" alt="2_0" src="https://github.com/user-attachments/assets/d888c0e0-116c-4c9e-9d2c-db1570ebe312" />
<img width="80%" alt="1_15800" src="https://github.com/user-attachments/assets/a7b9fa1a-cab4-436a-aa1a-008a345d9b11" />
<img width="80%" alt="1_14700" src="https://github.com/user-attachments/assets/480e4c7f-bc3e-4cdf-b0d6-6a51474a9d1b" />
</p>

## Requirements and Installation

We recommend running our code using:
* NVIDIA GPU + CUDA
* Python 3.9.2, Anaconda

To install our implementation, clone our repository and run following commands to install necessary packages:
   ```shell script
conda create -n stig-pgd python=3.9.2
conda activate stig-pgd
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
### Preparing Dataset
Put real and fake datasets in the folder ```datasets/```.

The real images should be located in ```datasets/{dataset_name}/real/```.<br>
The generated images should be located in ```datasets/{dataset_name}/fake/```.<br>
We suppose the type of image file is ```.png```.<br>
And its size is 256 * 256.<br>
The real and fake datasets must be prepared with an equal number of samples.<br>

We generated the AI images using the Stable Diffusion v1.5 Realistic Vision (SD 1.5 RV) model(https://github.com/lllyasviel/Fooocus).<br>
For the real dataset, we used cat images from https://www.kaggle.com/datasets/crawford/cat-dataset<br> and human images from https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq?select=00002.png.<br>
You can generate or acquire images using other Stable Diffusion models, and you should use real images that match the domain of those AI pictures.<br>


## Getting Started

Since the training process is time-consuming, a ```Demo``` has been prepared. You can perform inference and evaluation using the provided files with the detector. <br>
If you would like to know more about the training, inference methods, and the detector's learning and evaluation process, refer to the ```Training, Inference, and Detector Evaluation``` section. <br>

<details>
<summary> Demo </summary>

The dataset and checkpoint files required for demo execution have been prepared. Please use the Google Drive link below to download the files. <br>
[Demo file](https://drive.google.com/drive/folders/1q6-aDSPMqo_txCLIuriYCSQHHrsGAvaW?usp=sharing) <br>
If you have downloaded the ```Demo``` folder, you will find three subfolders.<br>
```datasets```: Contains folders for fake and real images, with 2,400 images in each. <br>
```model_checkpoint```: Contatins the pre-trained checkpoint file for STIG-PGD, which is used for inference. <br>
```pretrained_detectors```: Includes checkpoint folders for the ViT and DIF detectors. Within this folder, ```human_vit``` contains the checkpoint for the ViT, and ```human_dif``` contains the checkpoint for the DIF. These are used for detector evaluation. <br>
Executing ```gradio_app.py``` allows you to perform inference in the ```Inference``` tab and run detector evaluations in the ```Detect``` tab. <br>

### Inference

<p align="center">
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/f12bcdda-aa2a-49e5-a7d0-53062b6cb513" />
</p>

1. Enter the absolute path of the pre-trained STIG-PGD checkpoint file in the ```모델 체크포인트 경로```. <br>
2. Fix the ```GPU 디바이스``` and ```이미지 크기``` to their current values. <br>
3. Enter the absolute path of the ```fake``` folder inside the ```datasets``` folder in the ```입력 데이터셋 폴더 경로```. <br>
4. Enter the absolute path of the folder where the inference result images will be saved in the ```추론 결과 저장 폴더 경로```. The inference folder will be created automatically, so you don't need to create it.<br>
5. Click the ```추론 시작``` button. <br>

When inference is complete, a ```denoised``` folder containing the result images will be generated in the specified inference result directory. <br>

### Evaluation

<p align="center">
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/b6d351fb-eb17-41bb-9493-1ff711b50b0a" />
</p>

1. Select the detector model to be used for evaluation. <br>
2. Enter the absolute folder path of the detector model in the ```감지기 모델 폴더 경로```. (If you select ViT, please enter the absolute path of the ```human_vit``` folder. If you select DIF, please enter the absolute path of the ```human_dif``` folder. <br>
3. Enter the absolute folder path of the real images in the ```Real 이미지 데이터셋 폴더 경로```. (This corresponds to the ```real``` folder within the ```datasets``` folder.) <br>
4. Enter the absolute folder path of the fake image dataset in the ```Fake 이미지 데이터셋 폴더 경로```. (If you using the original fake images, enter the ```fake``` folder within the ```datasets``` folder. If using the fake images created after inference, enter the ```denoised``` folder within the inference resultd folder.) <br>
5. Click the ```평가 시작``` button. <br>

Once the evaluation is complete, the F1 score and Accuracy will be displayed. You can also view how the AI made its decisions for a randomly selected real and fake images. If the image is determined to be a genuine image, the "Real" label is output; if it is determined to be an AI-generated image, the "Fake" label is output. <br>

</details>

<details>
<summary> Training, Inference, and Detector Evaluation </summary>
    
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
python train_dif.py datasets/{dataset_name} checks/{checkpoint_name} -e {epochs} -f 1
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
    
</details>

## Reference
| Type | Title & Source | GitHub / Codebase |
| :--- | :--- | :--- |
| **STIG** (Base Method) | Lee, S., Jung, S. W., & Seo, H. (2024). **Spectrum translation for refinement of image generation (STIG) based on contrastive learning and spectral filter profile.** *In Proceedings of the AAAI Conference on Artificial Intelligence.* | [ykykyk112/STIG](https://github.com/ykykyk112/STIG) |
| **PGD** (Codebase) | Madry, A., et al. (2018). **Towards deep learning models resistant to adversarial attacks.** *In International Conference on Learning Representations (ICLR).* <br> (Implementation used for this project) | [Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch/tree/master) |
## Team Introduction
| Name | Student ID | Major |
| :--- | :--- | :--- |
| **Hyeonjun Cha** | 202011378 | Computer Science and Engineering |
| **Euntaek Lee** | 201911203 | Computer Science and Engineering |
| **Kyeongbeom Park** | 202011291 | Computer Science and Engineering |

