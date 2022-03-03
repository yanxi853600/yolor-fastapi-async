import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import io
import time
import datetime
from datetime import date

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from detection.utils.google_utils import attempt_load
from detection.utils.datasets import LoadStreams, LoadImages
from detection.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from detection.utils.plots import plot_one_box
from detection.utils.torch_utils import select_device, load_classifier, time_synchronized

from detection.models.models import *
from detection.utils.datasets import *
from detection.utils.general import *

import asyncio

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def load_model(opt, error_logging):
    try:
        # Initialize
        t0 = time.time()
        device_ = select_device(opt.device)
        half_ = device_.type != 'cpu'  # half precision only supported on CUDA
        
        # Load model
        model_ = Darknet(opt.cfg, opt.img_size).cuda()
        model_.load_state_dict(torch.load(opt.weights[0], map_location=device_)['model'])
        model_.to(device_).eval()
        if half_: model_.half()  # to FP16
        error_logging.info("--> ========== model loaded finished. (%.3fs) ========== " % (time.time() - t0))

    except Exception as e:
        error_logging.warning('  !!! model loaded failed : '+ str(e) )

    return model_, device_, half_

async def save_fig(loop, outputimg_path, im0, error_logging):
    try:
        t5 = time.time()
        await asyncio.sleep(0.01)
        cv2.imwrite(outputimg_path, im0)
        error_logging.info("--> save_fig finished. (%.3fs)" % (time.time() - t5))

    except Exception as e:
        error_logging.warning('  !!! save_fig failed : '+ str(e) )



def detect(opt, model, images, component_pad, device, half,filename, error_logging, save_img=False):
    out, source, view_img, save_txt, imgsz, names = \
        opt.output, images, opt.view_img, opt.save_txt, opt.img_size, opt.names

    fail_count = 0
    pass_count_equal2padnum = 0
    fail_count_passlessthanpadnum = 0
    fail_count_nolabel = 0

    save_img = True

    today=str(date.today())
    uploads_dir = os.getcwd() +'//'+ opt.output +'//'+str(today) + '//uploads//'
    bkp_input = os.getcwd() +'//'+ opt.output +'//'+str(today) + '//bkp_input//'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir, exist_ok=True)
        os.makedirs(bkp_input, exist_ok=True)
    else:
        pass

    try:
        save_img = Image.open(io.BytesIO(source))
        save_img.save(os.path.join(uploads_dir, filename))
    except Exception as e:
        print("err: "+str(e))

    try:
        im0s = np.array(Image.open(os.path.join(uploads_dir, filename)))
        
    except Exception as e:
        print("err: "+str(e))

    # dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    detect_anomal_Flag = False
    # Run inference
    try:
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # for path, img, im0s, vid_cap in dataset:
        # Padded resize
        img = letterbox(im0s, new_shape=imgsz, auto_size=imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(device) 
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) 
    except Exception as e:
        print("err: "+str(e))

    # Inference
    try:
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        error_logging.info("--> model prediction finished.(%.3fs)" % (t2 - t1))

    except Exception as e:
        error_logging.warning('  !!! model prediction is failed : '+ str(e) )

    list_label_image = []

    # Process detections
    try:
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            # else:
            p, s, im0 = os.path.join(uploads_dir, filename), '', im0s

            if not os.path.exists(str(Path(out) / today / 'results')):
                os.makedirs(str(Path(out) / today / 'results'), exist_ok=True)
            else:
                pass
            

            save_path = str(Path(out) / today / 'results')
            txt_path = str(Path(out) / Path(p).stem) 

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh tensor([160,  80, 160,  80])
            try:
                t3 = time_synchronized()
                if det is not None and len(det):
                    
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            list_label_image.append(names[int(cls)])
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                im0 = im0[:, :, ::-1]
                t4 = time_synchronized()
                error_logging.info("--> Plot bbox is finished.(%.3fs)" % (t4 - t3))

            except Exception as e:
                error_logging.warning('  !!! Plot bbox is failed : '+ str(e) )

            # Caculate the component_pad
            try:
                if "fail" in list_label_image:
                    save_img.save(os.path.join(bkp_input, filename))
                    fail_count += 1
                    output_path = save_path + '/fail/'
                    outputimg_path = output_path + Path(p).name.replace('.jpg','_fail.jpg')
                    detect_anomal_Flag = True # return fail
                elif list_label_image.count('pass') >= int(component_pad):
                    pass_count_equal2padnum += 1
                    output_path = save_path + '/pass/'
                    outputimg_path = output_path + Path(p).name.replace('.jpg','_pass.jpg')
                elif list_label_image.count('pass') < int(component_pad):
                    save_img.save(os.path.join(bkp_input, filename))
                    fail_count_passlessthanpadnum += 1
                    output_path = save_path + '/fail/'
                    outputimg_path = output_path + Path(p).name.replace('.jpg','_passless-than-pad_fail.jpg')
                    detect_anomal_Flag = True # return fail
                elif list_label_image.count('pass')==0:
                    save_img.save(os.path.join(bkp_input, filename))
                    fail_count_nolabel += 1
                    output_path = save_path + '/fail/' 
                    outputimg_path = output_path + Path(p).name.replace('.jpg','_nolabel_fail.jpg')
                    detect_anomal_Flag = True # return fail

            except Exception as e:
                error_logging.warning('  !!! Count label is failed : ' + str(e) )

            # Save results (image with detections)
            if save_img:
                os.makedirs(output_path, exist_ok=True)
                loop = asyncio.get_event_loop()
                loop.create_task(save_fig(loop, outputimg_path, im0, error_logging) )
        
        error_logging.info("--> Process detections finished.")

    except Exception as e:
        error_logging.warning('  !!! Process detections is failed : '+ str(e) )        

    # Caculate the component_pad
    error_logging.info('--> * fail_count: ' +str(fail_count))
    error_logging.info('--> * pass_count_equal2padnum: '+str(pass_count_equal2padnum))
    error_logging.info('--> * fail_count_passlessthanpadnum: '+str(fail_count_passlessthanpadnum))
    error_logging.info('--> * fail_count_nolabel: '+str(fail_count_nolabel))

    # Remove upload folder
    try:
        if os.path.exists(os.path.join(uploads_dir)):
            shutil.rmtree(os.path.join(uploads_dir))
        error_logging.info("--> Remove upload folder successfully.")
    except Exception as e:
        error_logging.warning('  !!! Remove upload folder failed : '+ str(e) )   

    return detect_anomal_Flag
