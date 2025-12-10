import torch
from model.cnn import eval_from_opt, train_from_opt, build_classifier_from_vit, evaluation, to_spectrum
from utils.util import OptionConfigurator, fix_randomseed
from dataclasses import dataclass
import threading
from typing import Optional, Generator, Tuple, List
import numpy as np
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
from trainer_dif import TrainerMultiple, distance
from dif_utils import *
from pathlib import Path
import data_dif 
from random import shuffle

NUM_NONE_TO_SEND = 20
blank_outputs = [None] * NUM_NONE_TO_SEND

@dataclass
class DetectConfig:
    model_path: str
    real_data_path: str
    fake_data_path: str
    device: int = 0
    size: int = 256
    classifier: str = 'vit'
    is_train: bool = False
    eval_root: str = 'none'
    lr: float = 0.0002
    batch_size: int = 1

class DetectRunner:
    def __init__(self):
        self.model = None
        self.device = None
        self.is_running = False
        self.should_stop = False
        self._lock = threading.Lock()
    
    def stop_detect(self):
        """추론을 중단합니다."""
        with self._lock:
            self.should_stop = True
    
    def run_detect10(
        self, 
        config: DetectConfig,
        progress_callback=None
    ):
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
            self.should_stop = False
        
        try:
            fix_randomseed(58)
            
            is_gpu = torch.cuda.is_available()
            self.device = torch.device(config.device) if is_gpu else torch.device('cpu')
            
            if config.classifier == 'vit':
            
                model = build_classifier_from_vit(config)
                param = torch.load(os.path.join(config.model_path, 'model.pt'), map_location = torch.device('cpu'))
                model.load_state_dict(param)
                model = model.to(self.device)

                datasets = ClassificationDataset(config)
                data_loader = datasets.only_real_top5(config)
                r_image_paths = datasets.img_list
                
                r_preds_label = []
                total_count = len(datasets.img_list)
                
                for idx, (data, target) in enumerate(data_loader):
                    
                    data, target = data.to(self.device), target.to(self.device)

                    data = to_spectrum(data)

                    with torch.no_grad():
                        output = model(data)
                        loss = model.criterion(output, target)
                        
                    _, pred = torch.max(output, dim=1)
                    
                    if pred == target.data:
                        r_preds_label.append("Real")
                    else:
                        r_preds_label.append("Fake")
                        
                
                
                data_loader = datasets.only_fake_top5(config)
                f_image_paths = datasets.img_list
                total_count = len(datasets.img_list) 
                
                f_preds_label = []
                
                for idx, (data, target) in enumerate(data_loader):
                    
                    data, target = data.to(self.device), target.to(self.device)

                    data = to_spectrum(data)

                    with torch.no_grad():
                        output = model(data)
                        loss = model.criterion(output, target)
                        
                    _, pred = torch.max(output, dim=1)
                    
                    if pred == target.data:
                        f_preds_label.append("Fake")
                    else:
                        f_preds_label.append("Real")
                
                return *r_image_paths, *r_preds_label, *f_image_paths, *f_preds_label
            
            elif config.classifier == 'dif':
                
                
                
                model_ep = 10
                
                checkpoints_dir = Path(config.model_path)
                fake_dir = Path(config.fake_data_path)
                real_dir = Path(config.real_data_path)
                
                with open(checkpoints_dir / "train_hypers.pt", 'rb') as pickle_file:
                    hyper_pars = pickle.load(pickle_file)
                    
                
                hyper_pars['Device'] = self.device
                hyper_pars['Batch Size'] = 1
                
                
                datasets = ClassificationDataset(config)
                data_loader = datasets.only_real_top5(config)
                r_image_paths = datasets.img_list
                r_preds_label = []
                
                
                
                trainer = TrainerMultiple(hyper_pars)
                trainer.load_stats(checkpoints_dir / f"chk_{model_ep}.pt")

                trainer.reset_test()
                trainer.calc_centers()
                
                fingerprint = trainer.fingerprint.to(self.device)
                fingerprint.repeat((config.batch_size, 1, 1, 1))
                
                with torch.no_grad():
                    for idx, (images, labels) in enumerate(data_loader):
                        
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        residuals = trainer.denoiser.denoise(images).float()

                        corr = trainer.corr_fun(fingerprint, residuals)
                        loss = trainer.loss_contrast(corr.mean((1, 2, 3)), labels) / trainer.m

                        corr = corr.mean((1, 2, 3))
                        
                        corr_np = corr.cpu().numpy()

                        # distance 함수를 사용하여 거리 계산 (calc_accuracy 로직 참조)
                        # dist.shape는 (배치 크기, 2)가 됩니다. (Real 거리, Fake 거리)
                        dist = distance(corr_np.reshape(-1, 1), trainer.mu_real, trainer.mu_fake)

                        # 3. 예측 레이블 (cls): 거리가 더 가까운 클러스터 (0 또는 1)
                        # np.argmin(axis=1)을 사용하여 [Real 거리, Fake 거리] 중 더 작은 인덱스(0=Real, 1=Fake)를 선택
                        preds = np.argmin(dist, axis=1) # 0 또는 1 (NumPy 배열)

                        # 4. 원본 레이블과 예측 레이블 비교 (일치하면 1, 다르면 0)
                        # np.equal은 요소별 비교를 수행합니다. True/False 배열 반환.
                        if preds == 0:
                            r_preds_label.append("Real")
                        else:
                            r_preds_label.append("Fake")

                
                data_loader = datasets.only_fake_top5(config)
                f_image_paths = datasets.img_list     
                f_preds_label = []
    
                trainer = TrainerMultiple(hyper_pars)
                trainer.load_stats(checkpoints_dir / f"chk_{model_ep}.pt") 

                trainer.reset_test()
                trainer.calc_centers()
                
                fingerprint = trainer.fingerprint.to(self.device)
                fingerprint.repeat((config.batch_size, 1, 1, 1))

                
                with torch.no_grad():
                    for idx, (images, labels) in enumerate(data_loader):
                        
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        residuals = trainer.denoiser.denoise(images).float()

                        corr = trainer.corr_fun(fingerprint, residuals)
                        loss = trainer.loss_contrast(corr.mean((1, 2, 3)), labels) / trainer.m

                        corr = corr.mean((1, 2, 3))
                        
                        corr_np = corr.cpu().numpy()
                        labels_np = labels.cpu().numpy()

                        # distance 함수를 사용하여 거리 계산 (calc_accuracy 로직 참조)
                        # dist.shape는 (배치 크기, 2)가 됩니다. (Real 거리, Fake 거리)
                        dist = distance(corr_np.reshape(-1, 1), trainer.mu_real, trainer.mu_fake)

                        # 3. 예측 레이블 (cls): 거리가 더 가까운 클러스터 (0 또는 1)
                        # np.argmin(axis=1)을 사용하여 [Real 거리, Fake 거리] 중 더 작은 인덱스(0=Real, 1=Fake)를 선택
                        preds = np.argmin(dist, axis=1) # 0 또는 1 (NumPy 배열)

                        # 4. 원본 레이블과 예측 레이블 비교 (일치하면 1, 다르면 0)
                        # np.equal은 요소별 비교를 수행합니다. True/False 배열 반환.
                        if preds == 1:
                            f_preds_label.append("Fake")
                        else:
                            f_preds_label.append("Real")
                
                
                
                return *r_image_paths, *r_preds_label, *f_image_paths, *f_preds_label
                        
        except Exception as e:
            return (*blank_outputs,)
            
        finally:
            with self._lock:
                self.is_running = False
                self.should_stop = False
        
        
    def run_detect(
        self, 
        config: DetectConfig,
        progress_callback=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, str, int, int], None, None]:
        
        with self._lock:
            if self.is_running:
                yield None, None, "이미 추론이 진행 중입니다.", 0, 0
                return
            self.is_running = True
            self.should_stop = False
        
        try:
            fix_randomseed(58)
            
            is_gpu = torch.cuda.is_available()
            self.device = torch.device(config.device) if is_gpu else torch.device('cpu')
            
            if config.classifier == 'vit':
            
                model = build_classifier_from_vit(config)
                param = torch.load(os.path.join(config.model_path, 'model.pt'), map_location = torch.device('cpu'))
                model.load_state_dict(param)
                model = model.to(self.device)

                datasets = ClassificationDataset(config)
                data_loader = datasets.get_loader()
                
                n_test_sample = 0
                test_acc, test_loss = 0., 0.
                
                all_preds = []
                all_targets = []
                
                total_count = len(datasets.img_list)
                
                model.eval()

                for idx, (data, target) in enumerate(data_loader):
                    # 중단 체크
                    with self._lock:
                        if self.should_stop:
                            yield None, None, f"추론 중단됨 ({idx}/{total_count})", idx, total_count
                            break
                    
                    
                    data, target = data.to(self.device), target.to(self.device)

                    data = to_spectrum(data)

                    with torch.no_grad():
                        output = model(data)
                        loss = model.criterion(output, target)

                    _, pred = torch.max(output, dim=1)
                    test_acc += torch.sum(pred == target.data).item()
                    test_loss += loss.item()
                    n_test_sample += data.shape[0]

                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    yield None, None, f"처리 중: {idx+1}/{total_count}", idx+1, total_count

                avg_test_acc = (test_acc / n_test_sample * 100.)
                avg_test_loss = (test_loss / n_test_sample * 100.)

                # Precision, Recall, F1-score 계산
                precision = precision_score(all_targets, all_preds, average='macro') * 100
                recall = recall_score(all_targets, all_preds, average='macro') * 100
                f1 = f1_score(all_targets, all_preds, average='macro') * 100

                yield f1, avg_test_acc, f"✅ 추론 완료!", total_count, total_count
                
            elif config.classifier == 'dif':
                
                
                model_ep = 10
                
                checkpoints_dir = Path(config.model_path)
                fake_dir = Path(config.fake_data_path)
                real_dir = Path(config.real_data_path)
                
                with open(checkpoints_dir / "train_hypers.pt", 'rb') as pickle_file:
                    hyper_pars = pickle.load(pickle_file)
                    
                
                hyper_pars['Device'] = self.device
                hyper_pars['Batch Size'] = 1
                
                real_path_list = [list((real_dir).glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
                real_path_list = [ele for ele in real_path_list if ele != []][0]

                fake_path_list = [list((fake_dir).glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
                fake_path_list = [ele for ele in fake_path_list if ele != []][0]
                
                test_set = data_dif.PRNUData(real_path_list, fake_path_list, hyper_pars, demand_equal=False,
                             train_mode=False)
                
                trainer = TrainerMultiple(hyper_pars)
                trainer.load_stats(checkpoints_dir / f"chk_{model_ep}.pt")
                
                test_loader = test_set.get_loader()

                trainer.reset_test()
                trainer.calc_centers()
                
                
                fingerprint = trainer.fingerprint.to(self.device)
                fingerprint.repeat((config.batch_size, 1, 1, 1))
                
                total_count = len(test_set)

                with torch.no_grad():
                    for idx, (images, labels) in enumerate(test_loader):
                        # 중단 체크
                        with self._lock:
                            if self.should_stop:
                                yield None, None, f"추론 중단됨 ({idx}/{total_count})", idx, total_count
                                break
                        
                        
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        residuals = trainer.denoiser.denoise(images).float()

                        corr = trainer.corr_fun(fingerprint, residuals)
                        loss = trainer.loss_contrast(corr.mean((1, 2, 3)), labels) / trainer.m

                        corr = corr.mean((1, 2, 3))

                        trainer.test_loss = trainer.test_loss + loss.tolist()
                        trainer.test_labels = trainer.test_labels + labels.tolist()

                        if trainer.test_corr_r is None:
                            trainer.test_corr_r = corr[~labels].cpu().numpy()
                            trainer.test_corr_f = corr[labels].cpu().numpy()
                        else:
                            trainer.test_corr_r = np.append(trainer.test_corr_r, corr[~labels].cpu().numpy(), axis=0)
                            trainer.test_corr_f = np.append(trainer.test_corr_f, corr[labels].cpu().numpy(), axis=0)
                        
                        yield None, None, f"처리 중: {idx+1}/{total_count}", idx+1, total_count
                                         
                    acc_f, acc_r = trainer.calc_accuracy(print_res=False)
                    n_fake = len(fake_path_list)
                    n_real = len(real_path_list)
                    
                    TP = acc_f * n_fake
                    FN = (1 - acc_f) * n_fake
                    FP = (1 - acc_r) * n_real
                    Precision = TP / (TP + FP)
                    Recall = TP / (TP + FN)
                    F1 = 2 * ((Precision * Recall) / (Precision + Recall)) * 100  # 백분율로 변환
                    Accuracy = 50 * (acc_r + acc_f)    
                    
                    yield F1, Accuracy, f"✅ 추론 완료!", total_count, total_count
                
        except Exception as e:
            yield None, None, f"❌ 추론 오류: {str(e)}", 0, 0
            
        finally:
            with self._lock:
                self.is_running = False
                self.should_stop = False
    
detect_runner = DetectRunner()

class ClassificationDataset(Dataset):

    def __init__(self, config) :

        self.init_options(config)
        self.set_options(config)
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        # 항상 3채널(RGB)로 변환
        self.image = Image.open(self.img_list[index]).convert('RGB')
        self.image = self.transform(self.image)
        self.label = self.label_list[index]

        return self.image, self.label

    def __len__(self) :
        return len(self.img_list)

    def init_options(self, config) :
        self.batch_size = config.batch_size
        self.size = config.size

    def set_options(self, config) :
        self.fake_list = glob(os.path.join(config.fake_data_path, '*.png'))
        self.fake_list.extend(glob(os.path.join(config.fake_data_path, '*.jpg')))
        
        self.fake_label_list = [0] * len(self.fake_list)

        self.real_list = glob(os.path.join(config.real_data_path, '*.png'))
        self.real_list.extend(glob(os.path.join(config.real_data_path, '*.jpg')))
        
        self.real_label_list = [1] * len(self.real_list)

        self.img_list = self.fake_list + self.real_list
        self.label_list = self.fake_label_list + self.real_label_list

    def get_loader(self) :

        dataloader = DataLoader(self, self.batch_size, shuffle = True)
        return dataloader
    
    def only_real_top5(self, config):
        real_only_list = glob(os.path.join(config.real_data_path, '*.png'))
        real_only_list.extend(glob(os.path.join(config.real_data_path, '*.jpg')))
        
        shuffle(real_only_list)
        
        self.img_list = real_only_list[:5]
        self.label_list = [1] * len(real_only_list[:5])
        
        dataloader = DataLoader(self, self.batch_size, shuffle = False) # 셔플 X
        return dataloader
    
    def only_fake_top5(self, config):
        fake_only_list = glob(os.path.join(config.fake_data_path, '*.png'))
        fake_only_list.extend(glob(os.path.join(config.fake_data_path, '*.jpg')))
        
        shuffle(fake_only_list)
        
        self.img_list = fake_only_list[:5]
        self.label_list = [0] * len(fake_only_list[:5])
        
        dataloader = DataLoader(self, self.batch_size, shuffle = False)
        return dataloader

def load_model_for_gui_detect(model_path: str, device: int = 0, size: int = 256) -> str:
    """GUI에서 모델을 로드하는 함수"""
    config = DetectConfig(
        model_path=model_path,
        real_data_path="",
        fake_data_path="",
        save_path="",
        device=0,
        size=256
    )
    return detect_runner.load_model(config)

def stop_detect_for_gui():
    """GUI에서 추론을 중단하는 함수"""
    detect_runner.stop_detect()
    return "🛑 탐지 중단 요청됨"