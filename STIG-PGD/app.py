"""
STIG-PGD Inference GUI
Gradio 기반 추론 인터페이스
"""

import gradio as gr
import numpy as np
import os
import torch
from inference import inference_runner, InferenceConfig, load_model_for_gui, stop_inference_for_gui


def create_app():
    """Gradio 앱을 생성합니다."""
    
    with gr.Blocks(title="STIG-PGD Inference") as app:
        
        # 헤더
        gr.Markdown("""
        # 🔬 STIG-PGD Inference
        **Spectral Transform for Image Generation - Projected Gradient Descent**
        """)
        
        with gr.Row():
            # 왼쪽 패널: 설정
            with gr.Column(scale=1):
                
                # 모델 설정 섹션
                gr.Markdown("### 📁 모델 설정")
                model_path = gr.Textbox(
                    label="모델 체크포인트 경로",
                    placeholder="예: ./checkpoints/parameters_100_epoch.pt",
                    value=""
                )
                
                with gr.Row():
                    device_input = gr.Number(
                        label="GPU 디바이스",
                        value=0,
                        precision=0,
                        minimum=0
                    )
                    size_input = gr.Number(
                        label="이미지 크기",
                        value=256,
                        precision=0,
                        minimum=64
                    )
                
                # 데이터 설정 섹션
                gr.Markdown("### 📂 데이터 설정")
                data_path = gr.Textbox(
                    label="입력 데이터 경로",
                    placeholder="예: ./datasets/inference/",
                    value=""
                )
                save_path = gr.Textbox(
                    label="저장 경로",
                    placeholder="예: ./results/inference_output/",
                    value="./results/inference_output"
                )
                
                # 실행 버튼
                gr.Markdown("### ▶️ 추론 실행")
                with gr.Row():
                    start_btn = gr.Button("▶️ 추론 시작", variant="primary")
                    stop_btn = gr.Button("⏹️ 중단", variant="stop")
                
                progress_text = gr.Textbox(
                    label="진행 상황",
                    value="대기 중...",
                    interactive=False
                )
                progress_bar = gr.Slider(
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
                        input_image = gr.Image(
                            label="입력",
                            type="numpy",
                            interactive=False
                        )
                    with gr.Column():
                        gr.Markdown("**📤 출력 이미지 (Denoised)**")
                        output_image = gr.Image(
                            label="출력",
                            type="numpy",
                            interactive=False
                        )
        
        # 푸터
        gr.Markdown("""
        ---
        *STIG-PGD: AI 생성 이미지의 주파수 도메인 기반 향상 모델*
        """)
        
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
        start_btn.click(
            fn=on_start_inference,
            inputs=[model_path, data_path, save_path, size_input, device_input],
            outputs=[input_image, output_image, progress_text, progress_bar]
        )
        
        stop_btn.click(
            fn=on_stop_inference,
            inputs=[],
            outputs=[progress_text]
        )
    
    return app


if __name__ == "__main__":
    # GPU 사용 가능 여부 확인
    print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA Device Count: {torch.cuda.device_count()}")
        print(f"🔧 CUDA Device Name: {torch.cuda.get_device_name(0)}")
    
    # 앱 생성 및 실행
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
