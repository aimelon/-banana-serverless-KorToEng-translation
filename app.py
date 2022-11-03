from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en", use_fast=False)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer
    global device

    # Parse out your arguments
    text = model_inputs.get('inputs', None)

    if text == None:
        return {'message': "No text provided"}
    
    encoded = tokenizer(text, return_tensors="pt")
    encoded.to(device)
    generated_tokens = model.generate(**encoded)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Return the results as a dictionary
    return {"translation": result[0].strip()}
