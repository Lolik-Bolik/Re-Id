import base64
import os
from urllib.parse import quote as urlquote
import dash_bootstrap_components as dbc
from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from inference import visualization, inference
from app import app
from app import server
import cv2 


weigths = {"custom": "/home/alexander/HSE_Stuff/Re-Id/log/model/model.pth.tar-150",
"pretrained": "/home/alexander/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"}
UPLOAD_DIRECTORY = "./app_uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# @server.route("/download/<path:path>")
# def download(path):
#     """Serve a file from the upload directory."""
#     return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

# /home/alexander/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth
class opts:
        data_path = "test_data"
        path_to_model = "/home/alexander/HSE_Stuff/Re-Id/log/model/model.pth.tar-150"
        path_to_custom_model = "/home/alexander/HSE_Stuff/Re-Id/log/model/model.pth.tar-150"
        path_to_pretrained_model = "/home/alexander/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
        batch_size = 1
opts = opts()



layout = html.Div(
    [   dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Github Link", href='https://github.com/Lolik-Bolik/Re-Id')),
            dbc.NavItem(dbc.NavLink("Upload image", href='/apps/app1')),
        ],
        brand="Person Re-Identification",
        brand_href="/",
        color="primary",
        dark=True,
    ),
        html.H2("Upload"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.H2("Result"),
        html.Div(
                [
                    html.H6("Weights"),
                    dcc.Dropdown(
                        id="method-name",
                        options=[{"label": i, "value": i} for i in weigths],
                        value="CuckooHashMap",
                    ),
                ],
                style={"width": "48%", "display": "inline-block"},
            ),
        html.Div(id='indicator-inference-result'),
        html.Div(id='indicator-inference-result-2')
    ],
    style={
        "background-image": 'url(/assets/background_2.jpg)',
        "background-repeat": "no-repeat",
        "background-position": "center",
        "background-size": "cover",
        "position": "fixed",
	    "min-height": "100%",
	    "min-width": "100%",}
)

def clear_uploaded_folder():
    for filename in os.listdir(UPLOAD_DIRECTORY):
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def save_file(weigths, name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    clear_uploaded_folder()
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))
    fig = inference(opts, weigths, is_plotly=True)
    return fig


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    Output("indicator-inference-result", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents"), Input("method-name", "value")],
)
def update_output(uploaded_filenames, uploaded_file_contents, weights):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            return dcc.Graph(figure=save_file(weights,name, data))

