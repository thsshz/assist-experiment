from flask import Flask
from flask import Response
from flask import request
import json


app = Flask(__name__)

dgpu_model_info = {
    "yoloV5n": {"lab1": "rtsp://172.16.29.105:5554/lab1"},
    "yoloV5s": {"lab1": "rtsp://172.16.29.105:5554/lab1"},
    "yoloV5m": {"lab1": "rtsp://172.16.29.105:5554/lab1"},
    "yoloV5l": {"lab1": "rtsp://172.16.29.105:5554/lab1"}
}

jetson_model_info = {
    "yoloV5n": {"lab1": "rtsp://172.16.29.105:5554/lab1"},
    "yoloV5s": {"lab1": "rtsp://172.16.29.105:5554/lab1"}
}

model_info = {
    "dgpu": dgpu_model_info,
    "jetson": jetson_model_info
}


@app.route("/api/camera_url", methods=["GET"])
def camera_url():
    """
        GET
        获取模型对应的分析摄像头列表
        params: deepstream-name
        return: code, info, data[{name, url}]
    """
    camera_data = {"code": 200, "data": []}
    try:
        name = request.args.get("name")
        server_name = request.args.get("server_name")
        cameras = model_info[server_name][name]
        for camera in cameras.keys():
            camera_data["data"].append({"name": camera, "url": cameras[camera]})
    except:
        return Response(json.dumps(camera_data), status=200, mimetype="application/json")
    return Response(json.dumps(camera_data), status=200, mimetype="application/json")