import streamlit as st
from PIL import Image
import requests

# st.set_page_config(layout="wide")

# cols for title and dropdown
col1, col2 = st.columns([3, 1])

with col1:
    st.title("Cells Segmentation")

with col2:
    model_versions = ["torch", "onnx"]
    selected_model = st.selectbox("Choose Model Version:", model_versions, index=0, key="model_select")

st.markdown("""<div style='text-align: center;'>
    <h3>Select an image for segmentation</h3>
</div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("""<div style='text-align: center;'>
        <h4>Uploaded Image</h4>
    </div>""", unsafe_allow_html=True)
    
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    api_url = f"http://localhost:8000/predict_{selected_model}"
    
    st.markdown("""<div style='text-align: center;'>
        <h4>Processing with Model: {}</h4>
    </div>""".format(selected_model), unsafe_allow_html=True)

    try:
        # send the image as a fastapi UploadFile
        response = requests.post(api_url, {"file": uploaded_file})

        if response.status_code == 200:
            st.success("Prediction successful!")
            segmented_image = Image.open(response.content)
            st.image(segmented_image, caption="Segmented Image", use_container_width=True)
        else:
            st.error(f"Error from API: {response.status_code}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
