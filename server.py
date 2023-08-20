from flask import Flask, request, jsonify
from solver import predict_captcha

app = Flask(__name__)

max_file_size = 1 * 1024 * 1024

app.config['MAX_CONTENT_LENGTH'] = max_file_size

@app.route('/solve', methods=['POST'])
def solve_captcha():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)

        if size > max_file_size: # 1MB
            return jsonify({'error': 'File size exceeds 1MB'}), 400
	
        try:
            predicted_text = predict_captcha(file.read())
            return predicted_text
        except Exception as e:
            return jsonify({'error': f'An error occurred processing the image'}), 500


if __name__ == '__main__':
    app.run(debug=False)
