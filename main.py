from asyncio.windows_events import NULL
import sys
from urllib3 import Timeout
import uvicorn
from enum import Enum
from typing import Optional
from Inference.errors import Error
from Inference.exceptions import ModelNotFound, InvalidModelConfiguration, ApplicationError, ModelNotLoaded, \
	InferenceEngineNotFound, InvalidInputData
from Inference.response import ApiResponse
from starlette.responses import FileResponse
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, File, UploadFile, Header, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import argparse
from detection import detect
from pydantic import BaseModel
from base64 import b64decode, b64encode
import numpy as np
from PIL import Image
import io
import sys
import base64
import cv2
from io import BytesIO
import logging
import logging.config
import os
import time
import datetime
from datetime import date
from detection.models import *
from detection.utils.datasets import *

from Inference.errors import Error
from requests.exceptions import Timeout
from traceback import extract_tb
from sys import exc_info,exit
import asyncio
#####################################################
# 	API Release Information (http://127.0.0.1:8888/docs)
#####################################################
app = FastAPI(version="1.0.0", title='Yolov3 inference Swagger',
			  description="<b>API for performing YOLOv3 inference.</b></br></br>"
						  "<b>Contact the developers:</b></br>"
						  "<b>Yanxi.Lin: <a href='mailto:Yanxi.Lin@advantech.com.tw'>Yanxi.Lin@advantech.com.tw</a></b></br>"
			 )
#####################################################
#	CORS Setting
#####################################################
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
	max_age=180,	#	timout (second)
)

#####################################################
#	Loaded yolov3 model
#####################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default="8888",help="port")
    parser.add_argument('--weights', nargs='+', type=str, default=['detection/weights/best_400.pt'], help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=448, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--cfg', type=str, default='detection/cfg/yolor_p6_smt_2c_448_coco.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='detection/cfg/SMT.names', help='*.cfg path')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    
    opt = parser.parse_args()
    return opt

opt = parse_args()
print(opt)
error_logging = Error()
model, device, half = detect.load_model(opt, error_logging)

#####################################################
#	Detect an image using yolov3
#####################################################
@app.post('/detect', tags=["POST Method"])
async def predict_image(component_pad : str = Form(...) , image: UploadFile = File(...)):
	try:
		t0 = time.time()
		images = await image.read()
		filename = image.filename
		results = ""

		# ============= Filter the wrong format of component_pad =============
		if component_pad.isnumeric()==True:
			error_logging.info("--> The number of pads are : " +str(component_pad))

			with torch.no_grad():
				label = detect.detect(opt, model, images, component_pad, device, half, filename, error_logging)
			
			if label == False:results = "pass"
				
			await asyncio.sleep(0.001)
			error_logging.info("--> ========== All Done. (%.3fs) ==========" % (time.time() - t0))
		else:
			error_logging.warning(" !!! component_pad has wrong format : " + str(component_pad)+" ( img_name: "+ filename +" ) ")
		# =====================================================================

		return Response(media_type="text/html", content=results)

	except Timeout as e :
		error_logging.warning('Timeout error'+str(e))
		return ApiResponse(success=False, error='Timeout error')	
	except ApplicationError as e:
		error_logging.warning(str(e))
		return ApiResponse(success=False, error=e)
	except Exception as e:
		error_logging.error('unexpected server error'+str(e))
		return ApiResponse(success=False, error='unexpected server error')
	

if __name__ == "__main__":
		uvicorn.run(app, host="0.0.0.0",port=opt.port, debug=True)


