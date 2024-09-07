import faiss
import requests
import numpy as np

d = 4000

titles = [
    "Echoes of the Future: Exploring Tomorrow's Innovations Today",
    "Whispers of the Past: A Journey Through Lost Civilizations",
    "The Art of Simple Living: Finding Joy in Minimalism",
    "Mysteries of the Mind: Unraveling the Secrets of Human Behavior",

]

index = faiss.IndexFlatL2(d)

X = np.zeros((len(titles), d), dtype='float32')

for i, title in enumerate(titles):
    res = requests.post('http://localhost:5000/api/embeddings',  # Note the corrected URL
                        json={
                            'model': 'llama3',
                            'prompt': title
                        })

embedding = res.json()['embedding']
X[i] = np.array(embedding)

index.add(X)

new_prompt = 'Fighter gets the ufc title'

res = requests.post('http://localhost:5000/api/embeddings',
                    json={
                        'model': 'llama3',
                        'prompt': new_prompt
                    })

embedding = np.array([res.json()['embedding']], dtype='float32')

D, I = index.search(embedding, k=5)

print(np.array(titles)[I.flatten()])
