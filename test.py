# from yolov9.detect import run

# run(weights='./models/640_v2.pt', source='./data/test', 
#     save_txt=True, save_conf=True,
#     project='./run/test', name='exp', half=True)

import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()