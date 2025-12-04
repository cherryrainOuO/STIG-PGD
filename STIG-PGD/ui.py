import gradio as gr
import numpy as np
import os
import torch
from inference import inference_runner, InferenceConfig, load_model_for_gui, stop_inference_for_gui


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
                            placeholder="예: ./checkpoints/parameters_100_epoch.pt",
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
                            label="입력 데이터 경로",
                            placeholder="예: ./datasets/inference/",
                            value=""
                        )
                        inf_save_path = gr.Textbox(
                            label="저장 경로",
                            placeholder="예: ./results/inference_output/",
                            value="./results/inference_output"
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
                gr.Markdown("## AI 감지기 관리")
                
                # 1. 학습 기능
                with gr.Accordion("학습", open=True):
                    gr.Markdown("### 모델 학습 설정")
                    
                    with gr.Row():
                        with gr.Column():
                            model_type = gr.Radio(
                                choices=["vit", "dif"],
                                value="vit",
                                label="모델 선택"
                            )
                            
                        with gr.Column():
                            epochs = gr.Number(
                                value=100,
                                label="Epoch 수",
                                precision=0
                            )
                    
                    destination_folder = gr.Textbox(
                        label="Destination 폴더명",
                        placeholder="예: my_model_output"
                    )
                    
                    train_button = gr.Button("학습 시작", variant="primary")
                    train_status = gr.Textbox(
                        label="학습 상태",
                        interactive=False
                    )
                
                gr.Markdown("---")
                
                # 2. 평가 기능
                with gr.Accordion("평가", open=True):
                    gr.Markdown("### 데이터셋 설정")
                    
                    with gr.Row():
                        with gr.Column():
                            real_dataset = gr.Textbox(
                                label="Real 이미지 데이터셋 경로",
                                placeholder="예: ./datasets/real/"
                            )
                        
                        with gr.Column():
                            fake_dataset = gr.Textbox(
                                label="Fake 이미지 데이터셋 경로",
                                placeholder="예: ./datasets/fake/"
                            )
                    
                    model_path = gr.Textbox(
                        label="모델 경로",
                        placeholder="예: ./model/checkpoint.pth"
                    )
                    
                    eval_button = gr.Button("평가 시작", variant="primary")
                    
                    gr.Markdown("### 평가 결과")
                    
                    # 평가 결과 10개 출력 공간
                    with gr.Row():
                        eval_results = gr.Dataframe(
                            headers=[
                                "Real 이미지", 
                                "Real 판독 결과", 
                                "Fake 이미지", 
                                "Fake 판독 결과", 
                                "보정된 Fake 이미지", 
                                "보정된 Fake 판독 결과"
                            ],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            row_count=10,
                            col_count=(6, "fixed"),
                            label="평가 결과 (10개)"
                        )
                    
                    gr.Markdown("### 전체 성능 지표")
                    
                    with gr.Row():
                        with gr.Column():
                            f1_score_output = gr.Textbox(
                                label="F1 Score",
                                interactive=False
                            )
                        
                        with gr.Column():
                            accuracy_output = gr.Textbox(
                                label="Accuracy",
                                interactive=False
                            )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()