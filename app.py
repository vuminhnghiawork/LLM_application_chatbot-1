from flask import Flask, render_template
from flask_cors import CORS		# newly added
from flask import request
import json
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

app = Flask(__name__)
CORS(app)				# newly added

# model_name = "facebook/blenderbot-400M-distill"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

nf4_config = BitsAndBytesConfig(
  load_in_4bit = True ,
  bnb_4bit_quant_type = "nf4",
  bnb_4bit_use_double_quant = True ,
  bnb_4bit_compute_dtype = torch.bfloat16
)

model_name = "ricepaper/vi-gemma-2-2b-function-calling"

# model_name = 'microsoft/Phi-3-mini-4k-instructtokenizer'
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = nf4_config, low_cpu_mem_usage = True)
conversation_history = []

@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

# First we need to define the structure of data in request from client
# data = {'prompt': 'message'}
@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    # Read prompt from HTTP request body
    data = request.get_data(as_text=True)   # Get data from post method
    data = json.loads(data) # Convert data to json
    input_text = data['prompt'] # Get message in data request

    # Create conversation history string
    history = "\n".join(conversation_history)

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")


    # Generate the response from the model
    outputs = model.generate(**inputs, max_length= 255)  # max_length will acuse model to crash at some point as history grows

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run(debug = True)
