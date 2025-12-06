import gradio as gr
import numpy as np
import os
import torch
from inference import inference_runner, InferenceConfig, load_model_for_gui, stop_inference_for_gui
from detect import detect_runner, DetectConfig, stop_detect_for_gui

NUM_NONE_TO_SEND = 20
blank_outputs = [None] * NUM_NONE_TO_SEND

def create_ui():
    with gr.Blocks(title="STIG-PGD") as app:
        gr.Markdown("# STIG-PGD Web Interface")
        
        with gr.Tabs():
            # Inference 탭
            with gr.Tab("Inference"):
                gr.Markdown("## 🔬 STIG-PGD Inference")
                
                with gr.Row():
                    # 왼쪽 패널: 설정
                    with gr.Column(scale=1):
                        # 모델 설정 섹션
                        gr.Markdown("### 📁 모델 설정")
                        inf_model_path = gr.Textbox(
                            label="모델 체크포인트 경로",
                            placeholder="예: C:/STIG-PGD/checkpoints/parameters_9_epoch.pt",
                            value=""
                        )
                        
                        with gr.Row():
                            inf_device_input = gr.Number(
                                label="GPU 디바이스",
                                value=0,
                                precision=0,
                                minimum=0
                            )
                            inf_size_input = gr.Number(
                                label="이미지 크기",
                                value=256,
                                precision=0,
                                minimum=64
                            )
                        
                        # 데이터 설정 섹션
                        gr.Markdown("### 📂 데이터 설정")
                        inf_data_path = gr.Textbox(
                            label="입력 데이터셋 폴더 경로",
                            placeholder="예: C:/STIG-PGD/datasets/inference/fake",
                            value=""
                        )
                        inf_save_path = gr.Textbox(
                            label="추론 결과 저장 폴더 경로",
                            placeholder="예: C:/STIG-PGD/results/inference_output",
                            value=""
                        )
                        
                        # 실행 버튼
                        gr.Markdown("### ▶️ 추론 실행")
                        with gr.Row():
                            inf_start_btn = gr.Button("▶️ 추론 시작", variant="primary")
                            inf_stop_btn = gr.Button("⏹️ 중단", variant="stop")
                        
                        inf_progress_text = gr.Textbox(
                            label="진행 상황",
                            value="대기 중...",
                            interactive=False
                        )
                        inf_progress_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="진행률 (%)",
                            interactive=False
                        )
                    
                    # 오른쪽 패널: 결과 출력
                    with gr.Column(scale=2):
                        gr.Markdown("### 🖼️ 추론 결과")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**📥 입력 이미지**")
                                inf_input_image = gr.Image(
                                    label="입력",
                                    type="numpy",
                                    interactive=False
                                )
                            with gr.Column():
                                gr.Markdown("**📤 출력 이미지 (Denoised)**")
                                inf_output_image = gr.Image(
                                    label="출력",
                                    type="numpy",
                                    interactive=False
                                )
                
                # 이벤트 핸들러
                def on_start_inference(model_path_val, data_path_val, save_path_val, size_val, device_val):
                    """추론 시작 버튼 클릭 핸들러 (제너레이터)"""
                    # 모델 경로 확인
                    if not model_path_val or not os.path.exists(model_path_val):
                        yield None, None, "❌ 유효한 모델 경로를 입력해주세요.", 0
                        return
                    
                    if not data_path_val or not os.path.exists(data_path_val):
                        yield None, None, "❌ 유효한 데이터 경로를 입력해주세요.", 0
                        return
                    
                    # 모델 로드
                    yield None, None, "🔄 모델 로딩 중...", 0
                    try:
                        load_result = load_model_for_gui(model_path_val, int(device_val), int(size_val))
                        if "❌" in load_result:
                            yield None, None, load_result, 0
                            return
                    except Exception as e:
                        yield None, None, f"❌ 모델 로드 실패: {str(e)}", 0
                        return
                    
                    # 설정 생성
                    config = InferenceConfig(
                        model_path=model_path_val,
                        data_path=data_path_val,
                        save_path=save_path_val,
                        size=int(size_val),
                        device=int(device_val)
                    )
                    
                    # 추론 실행
                    for input_img, output_img, status, current, total in inference_runner.run_inference(config):
                        progress = int((current / max(total, 1)) * 100) if total > 0 else 0
                        yield input_img, output_img, status, progress
                
                def on_stop_inference():
                    """추론 중단 버튼 클릭 핸들러"""
                    return stop_inference_for_gui()
                
                # 이벤트 연결
                inf_start_btn.click(
                    fn=on_start_inference,
                    inputs=[inf_model_path, inf_data_path, inf_save_path, inf_size_input, inf_device_input],
                    outputs=[inf_input_image, inf_output_image, inf_progress_text, inf_progress_bar]
                )
                
                inf_stop_btn.click(
                    fn=on_stop_inference,
                    inputs=[],
                    outputs=[inf_progress_text]
                )
            
            # Detect 탭
            with gr.Tab("Detect"):
                gr.Markdown("## 🔬 Fake-Image Detection")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        
                        
                        # 평가 기능
                        gr.Markdown("### 📁 모델 설정")
                        
                        with gr.Row():
                            with gr.Column():
                                eval_model_type = gr.Radio(
                                    choices=["vit", "dif"],
                                    value="vit",
                                    label="감지기 모델 선택"
                                )
                                
                            with gr.Column():
                                
                                eval_model_path = gr.Textbox(
                                    label="감지기 모델 폴더 경로",
                                    placeholder="예: C:/STIG-PGD/pretrained_detectors/vit",
                                    value=""
                                )
                        
                        # 데이터 설정 섹션
                        gr.Markdown("### 📂 데이터 설정")
                        
                        with gr.Row():
                            with gr.Column():
                                eval_real_dataset = gr.Textbox(
                                    label="Real 이미지 데이터셋 폴더 경로",
                                    placeholder="예: C:/STIG-PGD/datasets/inference/real",
                                    value=""
                                )
                            
                            with gr.Column():
                                eval_fake_dataset = gr.Textbox(
                                    label="Fake 이미지 데이터셋 폴더 경로",
                                    placeholder="예: C:/STIG-PGD/results/inference_output/denoised",
                                    value=""
                                )      
                        
                        # 실행 버튼
                        gr.Markdown("### ▶️ 평가 실행")
                        
                        with gr.Row():
                            eval_start_btn = gr.Button("평가 시작", variant="primary")
                            eval_stop_btn = gr.Button("중단", variant="stop")
                            
                        eval_detect_progress_text = gr.Textbox(
                            label="진행 상황",
                            value="대기 중...",
                            interactive=False
                        )
                        eval_detect_progress_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="진행률 (%)",
                            interactive=False
                        )
                        
                        gr.Markdown("### 📈 전체 성능 지표")
                    
                        with gr.Row():
                            with gr.Column(scale=1, min_width=150):
                                f1_score_output = gr.Textbox(
                                    label="F1 Score",
                                    value=0.0,
                                    interactive=False
                                )
                            
                            with gr.Column(scale=1, min_width=150):
                                accuracy_output = gr.Textbox(
                                    label="Accuracy",
                                    value=0.0,
                                    interactive=False
                                )
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 🖼️ 평가 결과")
                                
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**📥 Real 이미지**")
                                with gr.Row():
                                    r_image_1 = gr.Image(label="이미지 1", type="filepath") 
                                    r_label_1 = gr.Textbox(label="라벨 1", interactive=False)
                                with gr.Row():
                                    r_image_2 = gr.Image(label="이미지 2", type="filepath")
                                    r_label_2 = gr.Textbox(label="라벨 2", interactive=False)
                                with gr.Row():
                                    r_image_3 = gr.Image(label="이미지 3", type="filepath")                               
                                    r_label_3 = gr.Textbox(label="라벨 3", interactive=False)
                                with gr.Row():
                                    r_image_4 = gr.Image(label="이미지 4", type="filepath")                                
                                    r_label_4 = gr.Textbox(label="라벨 4", interactive=False)
                                with gr.Row():
                                    r_image_5 = gr.Image(label="이미지 5", type="filepath")                        
                                    r_label_5 = gr.Textbox(label="라벨 5", interactive=False
                                )
                            with gr.Column():
                                gr.Markdown("**📤 Fake 이미지**")
                                with gr.Row():
                                    f_image_1 = gr.Image(label="이미지 1", type="filepath") 
                                    f_label_1 = gr.Textbox(label="라벨 1", interactive=False)
                                with gr.Row():
                                    f_image_2 = gr.Image(label="이미지 2", type="filepath")
                                    f_label_2 = gr.Textbox(label="라벨 2", interactive=False)
                                with gr.Row():
                                    f_image_3 = gr.Image(label="이미지 3", type="filepath")                               
                                    f_label_3 = gr.Textbox(label="라벨 3", interactive=False)
                                with gr.Row():
                                    f_image_4 = gr.Image(label="이미지 4", type="filepath")                                
                                    f_label_4 = gr.Textbox(label="라벨 4", interactive=False)
                                with gr.Row():
                                    f_image_5 = gr.Image(label="이미지 5", type="filepath")                        
                                    f_label_5 = gr.Textbox(label="라벨 5", interactive=False
                                )
                                
                    
                    
                    
                    # 이벤트 핸들러
                def on_start_detect(eval_real_dataset, eval_fake_dataset, eval_model_type, eval_model_path):
                    """추론 시작 버튼 클릭 핸들러 (제너레이터)"""
                    # 모델 경로 확인
                    if not eval_real_dataset or not os.path.exists(eval_real_dataset):
                        yield None, None, "❌ 유효한 모델 경로를 입력해주세요.", 0, *blank_outputs 
                        return
                    
                    if not eval_fake_dataset or not os.path.exists(eval_fake_dataset):
                        yield None, None, "❌ 유효한 데이터 경로를 입력해주세요.", 0, *blank_outputs 
                        return
                    
                    if not eval_model_path or not os.path.exists(eval_model_path):
                        yield None, None, "❌ 유효한 데이터 경로를 입력해주세요.", 0, *blank_outputs 
                        return
                                  
                    # 설정 생성
                    config = DetectConfig(
                        real_data_path=eval_real_dataset,
                        fake_data_path=eval_fake_dataset,
                        classifier=eval_model_type,
                        model_path=eval_model_path
                    )
                    
                    
                    # 추론 실행
                    for f1, accuracy, status, current, total in detect_runner.run_detect(config):
                        progress = int((current / max(total, 1)) * 100) if total > 0 else 0
                        yield f1, accuracy, status, progress, *blank_outputs 
                    
                    test_10_outputs = detect_runner.run_detect10(config)
                    

                    yield f1, accuracy, status, progress, *test_10_outputs
                    
                    
                
                def on_stop_detect():
                    """추론 중단 버튼 클릭 핸들러"""
                    return stop_detect_for_gui()
                
                # 이벤트 연결
                eval_start_btn.click(
                    fn=on_start_detect,
                    inputs=[eval_real_dataset, eval_fake_dataset, eval_model_type, eval_model_path],
                    outputs=[f1_score_output, accuracy_output, eval_detect_progress_text, eval_detect_progress_bar,
                             r_image_1, r_image_2, r_image_3, r_image_4, r_image_5,
                             r_label_1, r_label_2, r_label_3, r_label_4, r_label_5,
                             f_image_1, f_image_2, f_image_3, f_image_4, f_image_5,
                             f_label_1, f_label_2, f_label_3, f_label_4, f_label_5]
                )
                
                eval_stop_btn.click(
                    fn=on_stop_detect,
                    inputs=[],
                    outputs=[eval_detect_progress_text]
                )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()