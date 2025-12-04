import torch
from utils.util import OptionConfigurator, fix_randomseed
from utils.dataset import get_inference_dataset_from_option, InferenceDataset
from model.stig_pgd_model import STIG_PGD
from tqdm import tqdm
import os
from utils.visualizing import Visualizer
from utils.log import Logger
from utils.metric import InferenceModel
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Generator, Tuple
import threading


def inverse_norm(tensor):
    return (tensor + 1.) * 0.5


@dataclass
class InferenceConfig:
    """추론 설정을 담는 데이터 클래스"""
    model_path: str
    data_path: str
    save_path: str
    size: int = 256
    batch_size: int = 1
    device: int = 0
    input_nc: int = 3
    output_nc: int = 3
    num_patch: int = 256
    sigma: int = 8
    norm: str = 'instance'
    nce_layers: str = '0,4,8,12,16'
    nce_idt: bool = True
    nce_T: float = 0.07
    lambda_GAN: float = 3.0
    lambda_NCE: float = 10.0
    lambda_PSD: float = 3.0
    lambda_LF: float = 3.0
    lambda_identity: float = 1.0
    lr: float = 0.00008
    beta1: float = 0.5
    beta2: float = 0.999


class InferenceRunner:
    """추론 실행을 관리하는 클래스"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.is_running = False
        self.should_stop = False
        self._lock = threading.Lock()
    
    def load_model(self, config: InferenceConfig) -> str:
        """모델을 로드합니다."""
        try:
            fix_randomseed(42)
            
            is_gpu = torch.cuda.is_available()
            self.device = torch.device(config.device) if is_gpu else torch.device('cpu')
            
            # config를 opt 형태로 변환
            opt = self._config_to_opt(config)
            
            self.model = STIG_PGD(opt, self.device).to(self.device)
            self.model.load_checkpoint(config.model_path)
            self.model.eval()
            
            return f"✅ 모델 로드 완료: {config.model_path}\n디바이스: {self.device}"
        except Exception as e:
            return f"❌ 모델 로드 실패: {str(e)}"
    
    def _config_to_opt(self, config: InferenceConfig):
        """InferenceConfig를 argparse Namespace 형태로 변환"""
        class Opt:
            pass
        
        opt = Opt()
        opt.size = config.size
        opt.batch_size = config.batch_size
        opt.device = config.device
        opt.input_nc = config.input_nc
        opt.output_nc = config.output_nc
        opt.num_patch = config.num_patch
        opt.sigma = config.sigma
        opt.norm = config.norm
        opt.nce_layers = config.nce_layers
        opt.nce_idt = config.nce_idt
        opt.nce_T = config.nce_T
        opt.lambda_GAN = config.lambda_GAN
        opt.lambda_NCE = config.lambda_NCE
        opt.lambda_PSD = config.lambda_PSD
        opt.lambda_LF = config.lambda_LF
        opt.lambda_identity = config.lambda_identity
        opt.lr = config.lr
        opt.beta1 = config.beta1
        opt.beta2 = config.beta2
        opt.inference_data = config.data_path
        opt.inference_params = config.model_path
        
        return opt
    
    def stop_inference(self):
        """추론을 중단합니다."""
        with self._lock:
            self.should_stop = True
    
    def run_inference(
        self, 
        config: InferenceConfig,
        progress_callback=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, str, int, int], None, None]:
        """
        추론을 실행하고 결과를 yield합니다.
        
        Yields:
            Tuple[input_image, output_image, save_path, current_idx, total_count]
        """
        with self._lock:
            if self.is_running:
                yield None, None, "이미 추론이 진행 중입니다.", 0, 0
                return
            self.is_running = True
            self.should_stop = False
        
        try:
            if self.model is None:
                yield None, None, "모델이 로드되지 않았습니다.", 0, 0
                return
            
            # 데이터셋 준비
            transform = transforms.Compose([
                transforms.Resize((config.size, config.size)),
                transforms.ToTensor(),
            ])
            
            # 이미지 파일 목록 가져오기
            from glob import glob
            img_list = glob(os.path.join(config.data_path, '*.png'))
            img_list.extend(glob(os.path.join(config.data_path, '*.jpg')))
            img_list.extend(glob(os.path.join(config.data_path, '*.jpeg')))
            
            if not img_list:
                yield None, None, f"데이터 경로에 이미지가 없습니다: {config.data_path}", 0, 0
                return
            
            # 저장 경로 생성
            os.makedirs(config.save_path, exist_ok=True)
            denoised_path = os.path.join(config.save_path, 'denoised')
            denoised_mag_path = os.path.join(config.save_path, 'denoised_mag')
            os.makedirs(denoised_path, exist_ok=True)
            os.makedirs(denoised_mag_path, exist_ok=True)
            
            total_count = len(img_list)
            
            for idx, img_path in enumerate(img_list):
                # 중단 체크
                with self._lock:
                    if self.should_stop:
                        yield None, None, f"추론 중단됨 ({idx}/{total_count})", idx, total_count
                        break
                
                # 이미지 로드 및 전처리
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # 추론 실행
                with torch.no_grad():
                    self.model.set_input(input_tensor, evaluation=True)
                    self.model.evaluation()
                
                # 결과 추출
                input_img = self.model.input_image_normed.detach().squeeze(0)
                denoised_img = self.model.denoised_image_normed.detach().squeeze(0)
                denoised_mag = self.model.denoised_mag.detach().squeeze(0).mean(0)
                
                # numpy로 변환
                input_np = np.transpose(input_img.cpu().numpy(), (1, 2, 0))
                denoised_np = np.transpose(denoised_img.cpu().numpy(), (1, 2, 0))
                denoised_mag_np = np.clip(denoised_mag.cpu().numpy(), 0.0, 1.0)
                
                # 이미지 저장
                save_name = f'{idx:06d}.png'
                plt.imsave(os.path.join(denoised_path, save_name), denoised_np)
                plt.imsave(os.path.join(denoised_mag_path, save_name), denoised_mag_np, cmap='jet')
                
                yield input_np, denoised_np, f"처리 중: {idx+1}/{total_count}", idx+1, total_count
            
            yield None, None, f"✅ 추론 완료! 저장 경로: {config.save_path}", total_count, total_count
            
        except Exception as e:
            yield None, None, f"❌ 추론 오류: {str(e)}", 0, 0
        finally:
            with self._lock:
                self.is_running = False
                self.should_stop = False


# 전역 추론 러너 인스턴스
inference_runner = InferenceRunner()


def load_model_for_gui(model_path: str, device: int = 0, size: int = 256) -> str:
    """GUI에서 모델을 로드하는 함수"""
    config = InferenceConfig(
        model_path=model_path,
        data_path="",
        save_path="",
        device=device,
        size=size
    )
    return inference_runner.load_model(config)


def run_inference_for_gui(
    data_path: str,
    save_path: str,
    size: int = 256,
    device: int = 0
):
    """GUI에서 추론을 실행하는 제너레이터 함수"""
    config = InferenceConfig(
        model_path="",  # 이미 로드됨
        data_path=data_path,
        save_path=save_path,
        size=size,
        device=device
    )
    
    for result in inference_runner.run_inference(config):
        yield result


def stop_inference_for_gui():
    """GUI에서 추론을 중단하는 함수"""
    inference_runner.stop_inference()
    return "🛑 추론 중단 요청됨"


if __name__ == '__main__':
    # CLI 모드로 실행
    fix_randomseed(42)
    
    opt = OptionConfigurator().parse_options()
    loader = get_inference_dataset_from_option(opt)

    is_gpu = torch.cuda.is_available()
    device = torch.device(opt.device) if is_gpu else torch.device('cpu')

    model = STIG_PGD(opt, device).to(device)
    model.load_checkpoint(opt.inference_params)

    save_path = os.path.join('./results', opt.dst)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'inference'), exist_ok=True)
    inferencer = InferenceModel(opt, os.path.join(save_path, 'inference'))

    for n, sample in enumerate(tqdm(loader, desc="{:17s}".format('Inference State'), mininterval=0.0001)):
        model.set_input(sample, evaluation=True)
        model.evaluation()
        inferencer.step(model, sample, n)
        

