from app import app
from flask import render_template
import rnn

print("Starting RNN")
RNN = rnn.GruRNN(
    hid_size=200,
    trunc=100,
    emb_size=100,
    dataset="datasets/all_stories.pkl"
)
print("Started RNN")

@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'Terry'}
    posts = [" ".join(RNN.generate_story()[0]) for i in range(5)]
    return render_template('index.html', title="Home", user=user, posts=posts)
