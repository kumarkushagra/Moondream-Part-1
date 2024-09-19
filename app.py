from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer

app = Flask(__name__)

# Load tokenizer and first part of the model
tokenizer = AutoTokenizer.from_pretrained("moondream/moondream2")
model_part1 = torch.load("moondream_part1.pth")

@app.route('/part1', methods=['POST'])
def run_part1():
    data = request.json['input']
    input_ids = tokenizer(data, return_tensors="pt")['input_ids']
    with torch.no_grad():
        part1_output = model_part1(input_ids)

    # Forward the output to Part 2 API
    part2_url = "https://<part2-deployment-url>/part2"
    response = requests.post(part2_url, json={'input': part1_output.tolist()})
    final_output = response.json()
    return jsonify({'output': final_output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
