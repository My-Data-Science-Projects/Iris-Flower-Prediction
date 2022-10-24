import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.tree import DecisionTreeClassifier

st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

image = Image.open('iris.png')
st.image(image, caption='Iris Flower Types')

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

iris = pd.read_csv('iris.csv')
X = iris.drop(columns=['species'])
Y = iris['species']

clf = DecisionTreeClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# st.write(iris['species'].unique())

st.subheader('Prediction Probability')
st.write(pd.DataFrame({
    'setosa': [str("{0:.2f}".format(prediction_proba[0][0]*100)) + " %"],
    'versicolor': [str("{0:.2f}".format(prediction_proba[0][1]*100)) + " %"],
    'virginica': [str("{0:.2f}".format(prediction_proba[0][2]*100)) + " %"]
}))
# print(prediction_proba[0][0])

st.subheader('Prediction')
# print(prediction)
# st.success(prediction[0])
st.markdown(f'<h1 style="width: 150px; text-align:center; color:#ffffff; font-size:24px; background-color:#239B56; padding:5px; border-radius:10px; font-weight:400; font-family:sans-serif">{prediction[0]}</h1>', unsafe_allow_html=True)
