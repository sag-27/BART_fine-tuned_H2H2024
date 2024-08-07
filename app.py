from flask import Flask, request, jsonify, render_template
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import signal

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = BartForConditionalGeneration.from_pretrained("C:/Users/Sagnik Sen/Desktop/HACK2HIRE/fine-tuned-bart")
tokenizer = BartTokenizer.from_pretrained("C:/Users/Sagnik Sen/Desktop/HACK2HIRE/fine-tuned-bart")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.json['input_text']
    
    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate response
    output_ids = model.generate(
        input_ids,
        max_length=150,
        num_beams=5,
        do_sample=True,         
        temperature=0.7,
        top_p=0.9,
        early_stopping=True
    )
    
    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return jsonify({"response": output_text})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
