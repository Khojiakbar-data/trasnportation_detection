import streamlit as st # type: ignore
from PIL import Image # type: ignore
from fastai.vision.all import * # type: ignore
import plotly.express as px # type: ignore
# Title
st.title("Classification Report for Transport")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Reset the file pointer to the beginning
        uploaded_file.seek(0)
        
        # Open the image using P
        # IL
        image = Image.open(uploaded_file)
        
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert the uploaded file to fastai's PILImage
        uploaded_file.seek(0)  # Ensure the file pointer is at the beginning again
        img = PILImage.create(uploaded_file) # type: ignore
        
        # Load the model
        model_path = '/Users/mak/Desktop/course_2023/DataScience/TransportClassification-1-main/transport_model.pkl'
        model = load_learner(model_path) # type: ignore
        
        # Predict the image
        pred_class, pred_idx, outputs = model.predict(img)
        
        # Display prediction
        st.write(f"Prediction: {pred_class}")
        st.write(f"Prediction Probability: {outputs[pred_idx]*100:.1f}%")
        fig=px.bar(x=outputs*100,y=model.dls.vocab)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    #No file uploaded
    st.write("No file uploaded")
