
import requests


def post_stream(img_path, URL="http://127.0.0.1:5000/object-tracking/push_stream"):
    # defining a params dict for the parameters to be sent to the API
    PARAMS = {
        "img_src": img_path
    }
    # this will make the method "POST"
    # post_data(URL, PARAMS)

    # sending get request and saving the response as response object
    requests.post(url=URL, params=PARAMS)
