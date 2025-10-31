import streamlit as st
import pickle
import re
import subprocess
import sys

# ✅ Auto-install nltk if missing (prevents "ModuleNotFoundError: No module named 'nltk'")
try:
    import nltk
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk

# ✅ Download stopwords only if not already present
try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

if __name__ == '__main__':
    st.title('Fake News Classification App')
    st.subheader("Input the News content below")

    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("Predict")

    if predict_btt:
        prediction_class = fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Reliable')
        elif prediction_class == [1]:
            st.warning('Unreliable')
