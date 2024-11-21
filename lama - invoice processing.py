!pip install git+https://github.com/huggingface/transformers
!pip install -q gradio

import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import requests

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

def extract_invoice_data(image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": """You are an expert in invoice data extraction.
                Extract data from the provided invoice images, based on the individual items and
                format the output as JSON."""
                }
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to GPU if available
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    output_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text[0]


    iface = gr.Interface(
    fn=extract_invoice_data,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Extracted Invoice Data", lines=30, max_lines=50),  # Increase lines for larger display
    title="Invoice Data Extraction",
    description="Upload an invoice image to extract and format the data as JSON.",
)

# Launch the app
iface.launch(share=True, debug=True)