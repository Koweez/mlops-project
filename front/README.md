## Front

Simple web page using Streamlit to display the results of the model. Front is running on port 8501.

### How to use ?

Build the docker image using the following command:
```bash
docker build -t streamlit .
```

Run the docker container using the following command:
```bash
docker run -p 8501:8501 streamlit
```

You can now access the front on http://localhost:8501
