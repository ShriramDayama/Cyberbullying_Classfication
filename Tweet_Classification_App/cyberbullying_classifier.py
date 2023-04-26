import re
import nltk
import pickle
import streamlit as st



def preprocess(string):
  pattern = "[#|@][^\s]*|https://[^\s]|www\.[^\s]"


  emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+"
    ,flags = re.UNICODE,
  )

  
  stop_words = set(nltk.corpus.stopwords.words('english'))
  
  string = re.sub(pattern , "", string)
  string = re.sub(emoji_pattern, "", string)

  tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
  string = tokenizer.tokenize(string)

  lemmatizer = nltk.stem.WordNetLemmatizer()
  string = [lemmatizer.lemmatize(word) for word in string]
  
  string = ' '.join(string)
  return string


predict_map = {0 : 'not_cyberbullying' , 
               1 :'religion',
               2 :'age',
               3 :'gender',
               4 :'ethnicity',
            }

model = pickle.load(open('cyberbullying-classifier.pkl','rb'))


def main():
  st.title('Cyberbullying Classifier')
  input = st.text_input('Tweet')

  processed_input = preprocess(input)

  if st.button('Classify Tweet'):
    
    result = model.predict([processed_input])

    st.success(predict_map[result[0]])


if __name__ == "__main__":
  main()
