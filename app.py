import streamlit as st
import requests as r
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import re 

st.title('Spam Detection  App')
st.write('*Note: it will take up to 30 seconds to run the app.*')
form = st.form(key='message-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')

#load model data
url = "models/spam_classifier.joblib"
#f = "C:/Users/rzouga/Desktop/ALLINHERE/ALLINHERE/FraudDetection/DeployPipeComplet/models/pipeline_model_lgbm_final.joblib"
# download model from Dropbox, cache it and load the model into the app
@st.cache(allow_output_mutation=True)
def load_model(url):
    model = joblib.load(url)
    return model   
# Preprocess Heleper 
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

if submit:
    fig, ax = plt.subplots()
    model = load_model(url)
    message = preprocessor(user_input)
    label = model.predict([message])[0]
    score = model.predict_proba([message])[0][1]
    if label == 'ham':
        st.success(f'This is a {label} : (score: {score})')
    else:
        st.error(f'OOPS it is a {label} : (score: {score})')
        
    classes = {0:'ham',1:'spam'}
    class_labels = list(classes.values())

    st.write("The predicted class is ",label)
    prob_ham= 1-score
    prob_spam = score
    probs = [prob_ham,prob_spam]
    ax = sns.barplot(probs ,class_labels, palette="winter", orient='h')
    ax.set_yticklabels(class_labels,rotation=0)
    plt.title("Probabilities of the Data belonging to each class")
    for index, value in enumerate(probs):
        plt.text(value, index,str(value))
    st.pyplot(fig) 
