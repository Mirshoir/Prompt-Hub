from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


@app.route('/api/embeddings', methods=['POST'])
def embedding():
    embedding = np.random.rand(4000).tolist()
    print("Generated embedding:", embedding)  # Debugging: Print generated embedding
    return jsonify({'embedding': embedding})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
