import gradio as gr
from transformers import pipeline
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch

# ---- Installations 
'''
pip install huggingface_hub
pip install sentencepiece
'''

# ---- Load model + tokenizer (reuse what you have)
# MBART 
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- Translation function
def translate(text):
    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["pt_XX"]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# ---- Gradio Interface
iface = gr.Interface(
    fn=translate,                   # function to run
    inputs=gr.Textbox(label="English"),   # user input
    outputs=gr.Textbox(label="Portuguese"), # translated output
    title="English → Portuguese Translator",
    description="Enter an English sentence and get the Portuguese translation using MBART."
)

# Launch the interface
iface.launch()

'''
# Set up
Create Hugging Face account - need to login through terminal with access token 

# Flow: 
Flow is tokenize input -> generate output tokens in target -> decode back to text (in Portuguese)

'''
