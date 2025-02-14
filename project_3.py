import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle
import streamlit as st

# Load the trained model
#with open('spam_ham_model.pkl', 'rb') as f:
    #model = pickle.load(f)

st.title('Spam/Ham Message Detector')

# User input
user_input = st.text_area("Enter a message to classify:")
model = pickle.load(open(r"C:\Users\u sudharshan\Downloads\forage_projects\spam_ham_model.pkl","rb")) #pickle file path
if st.button('Classify'):
    if user_input:
        # Preprocess the input
        #processed_input = preprocess_text(user_input)
        
        # Make prediction
        prediction = model.predict([user_input])[0]
        probability = model.predict_proba([user_input])[0]
        
        # Display result
        st.write(f"Classification: {prediction}")
        st.write(f"Confidence: {max(probability):.2f}")
        
        # Display probability distribution
        prob_df = pd.DataFrame({
            'Class': ['Ham', 'Spam'],
            'Probability': probability
        })
        st.bar_chart(prob_df.set_index('Class'))
    else:
        st.write("Please enter a message to classify.")

# Display some information about the model
st.sidebar.header("About the Model")
st.sidebar.write("This spam/ham detector uses a Naive Bayes classifier trained on a dataset of labeled messages.")
st.sidebar.write("The model preprocesses the text by removing stopwords, stemming, and vectorizing the words.")

# You can add more information or visualizations in the sidebar if needed