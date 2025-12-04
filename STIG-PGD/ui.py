import gradio as gr


def create_ui():
    with gr.Blocks(title="STIG-PGD") as app:
        gr.Markdown("# STIG-PGD Web Interface")
        
        with gr.Tabs():
            # Inference 탭 (빈 탭)
            with gr.Tab("Inference"):
                gr.Markdown("### Inference 기능")
                gr.Markdown("추후 구현 예정")
            
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
                            real_dataset = gr.File(
                                label="Real 이미지 데이터셋",
                                file_count="directory"
                            )
                        
                        with gr.Column():
                            fake_dataset = gr.File(
                                label="Fake 이미지 데이터셋",
                                file_count="directory"
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