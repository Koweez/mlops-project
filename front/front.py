import os
from io import BytesIO

import requests
import streamlit as st
from fastapi import UploadFile
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


async def convert_to_upload_file(uploaded_file):
    return UploadFile(
        filename=uploaded_file.name,
        file=BytesIO(uploaded_file.read()),
    )


model_to_select = requests.get(f"{BACKEND_URL}/get_available_models")

# print(model_to_select)

# wide layout
st.set_page_config(layout="wide")

# columns for title and dropdown
col1, col2 = st.columns([3, 1])

with col1:
    st.title("Cells Segmentation")

with col2:
    selected_model = st.selectbox(
        "Select a model", [model["name"] for model in model_to_select.json()]
    )

st.markdown(
    """<div style='text-align: center;'>
    <h3>Select an image for segmentation</h3>
</div>""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1_body, col2_body = st.columns([2, 2])

    with col1_body:
        st.markdown(
            """<div style='text-align: center;'>
            <h4>Uploaded Image</h4>
        </div>""",
            unsafe_allow_html=True,
        )
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    api_url = f"{BACKEND_URL}/predict_{selected_model}"

    with col2_body:
        st.markdown(
            f"""<div style='text-align: center;'>
            <h4>Processing with Model: {selected_model}</h4>
        </div>""",
            unsafe_allow_html=True,
        )

        try:
            body = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                ),
                "model_name": selected_model,
            }
            response = requests.post(api_url, files=body)

            if response.status_code == 200:
                # st.success("Prediction successful!")
                # received bytes
                # convert to image
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Segmented Image", use_container_width=True)

            else:
                st.error(f"Error from API: {response.status_code}")
                st.error(response.text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
