import os
import cv2
import mmcv
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as maskUtils
from argparse import ArgumentParser
from tqdm import tqdm
from mmdet.apis import inference_detector, init_detector
import pandas as pd
import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50V2, ResNet101V2
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import Xception

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json
import time

class simlarClothing():
    def __init__(self):
        self.seg_top_db=[]
        self.seg_top_db_file=[]
        self.seg_bottom_db=[]
        self.seg_bottom_db_file=[]
        self.seg_dress_db=[]
        self.seg_dress_db_file=[]
        
        self.database_top_encodings = []
        self.database_bottom_encodings = []
        self.database_dress_encodings = []
        self.database_img = []
        self.database_file = []
        self.model_path = '../data/model/segmentation.pth'
        self.config_file_path = '../data/model/mask_rcnn_r50_fpn_1x.py'
        self.input_path = '../data/img/'
        self.model_1 = init_detector(self.config_file_path, self.model_path, device='cuda:0')
        self.preprocess_input, self.input_size = self.get_pressprocess_info('ResNet50')
        self.model_2 = self.build_model('ResNet50')
        self.ec_data=pd.read_csv('../data/ec_data.csv')
        self.get_seg_database()
        self.get_feature()
        
        print("simlar_clothing init done.")
        
    def show_result(self, img, result, class_names, score_thr=0.8):
        bbox_result, segm_result = result

        ###########################################
        # for one class, only keep one bbox and one segm with highest confidence
        new_bbox_result = []
        new_segm_result = []
        for bbox, segm in zip(bbox_result, segm_result):
            if len(bbox) <= 1:
                new_bbox_result.append(bbox)
                new_segm_result.append(segm)
                continue
            max_ind = np.argmax(bbox[:, -1])
            new_segm_result.append(np.asarray([segm[max_ind]]))
            new_bbox_result.append(np.asarray([bbox[max_ind]]))

        bbox_result = new_bbox_result
        segm_result = new_segm_result
        #########################################

        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        label_candidate = [class_names[i] for i in labels]

        output_segmentation = {}
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            try:
                color_masks = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(max(labels) + 1)]

                for i in inds:
                    color_mask = color_masks[labels[i]]
                    mask = maskUtils.decode(segms[i]).astype(np.bool)

                    item = img.copy()
                    item[~mask] = 0

                    output_segmentation[label_candidate[i]] = item
            except:
                pass

        return output_segmentation            
    def get_seg_database(self):
        input_img_list = [file for file in os.listdir(self.input_path) if ('.jpg' in file) or ('.jpeg' in file) or ('.png' in file)]
        if input_img_list == []:
            print('Can not find any image in folder "inputs"...')
        for file in tqdm(input_img_list):
            img_path = os.path.join(self.input_path, file)

            # Read image
            img = cv2.imread(img_path)

            # Inference
            result = inference_detector(self.model_1, img)

            # Draw bbox and segmentarion mask
            output_segmentation = self.show_result(img, result, self.model_1.CLASSES, score_thr=0.6)

            for cate, seg_item in zip(output_segmentation.keys(), output_segmentation.values()):
                csv_cate = self.read_cate(img_path)

                if cate == 'top':
                    if csv_cate == 'top':
                        plot_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                        self.seg_top_db.append(plot_item) #記圖
                        self.seg_top_db_file.append(img_path) #記檔名
                elif cate == 'skirt' or cate == 'leggings' or cate == 'pants':
                    if csv_cate == 'bottom':
                        plot_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                        self.seg_bottom_db.append(plot_item)
                        self.seg_bottom_db_file.append(img_path)
                elif cate == 'dress':
                    if csv_cate == 'dress':
                        plot_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                        self.seg_dress_db.append(plot_item)
                        self.seg_dress_db_file.append(img_path)


                # if cate == 'top':
                #     plot_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                #     self.seg_database[img_path] = plot_item
                # elif cate == 'skirt' or cate == 'leggings' or cate == 'pants':
                #     plot_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                #     self.seg_database[img_path] = plot_item
                # elif cate == 'dress':
                #     plot_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                #     self.seg_database[img_path] = plot_item
                    
                # if csv_cate == 'top':
                #     self.seg_top_db.append(plot_item) #記圖
                #     self.seg_top_db_file.append(img_path) #記檔名
                # elif csv_cate == 'bottom':
                #     self.seg_bottom_db.append(plot_item)
                #     self.seg_bottom_db_file.append(img_path)
                # elif csv_cate == 'dress':
                #     self.seg_dress_db.append(plot_item)
                #     self.seg_dress_db_file.append(img_path)
                    
    def build_model(self,model_type):
        print('build_model ...')
        if  model_type == 'InceptionResNet':
            model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299,299,3))
        elif model_type == 'ResNet50V2':
            model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224,224,3))
        elif model_type == 'ResNet50':
            model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
        elif model_type == 'ResNet101V2':
            model = ResNet101V2(include_top=False, weights='imagenet', input_shape=(224,224,3))
        elif model_type == 'DenseNet':
            model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3))
        elif model_type == 'Xception':
            model = Xception(include_top=False, weights='imagenet', input_shape=(299,299,3))
        else:
            sys.exit()

        model.trainable = False
        output = keras.layers.GlobalAveragePooling2D()(model.output)
        model = keras.models.Model(name=model_type, inputs=model.inputs, outputs=output)
        return model

    ''' 載入圖像前處理資訊 '''
    def get_pressprocess_info(self,model_type): 
        if  model_type == 'InceptionResNet':
            preprocess_input = keras.applications.inception_resnet_v2.preprocess_input
            img_size = (299, 299)
        elif (model_type == 'ResNet50'):
            preprocess_input = keras.applications.resnet.preprocess_input
            img_size = (224, 224)
        elif (model_type == 'ResNet50V2'or model_type == 'ResNet101V2'):
            preprocess_input = keras.applications.resnet_v2.preprocess_input
            img_size = (224, 224)
        elif model_type == 'DenseNet':
            preprocess_input = keras.applications.densenet.preprocess_input
            img_size = (224, 224)
        elif model_type == 'Xception':
            preprocess_input = keras.applications.xception.preprocess_input
            img_size = (299, 299)
        return preprocess_input, img_size
    
    def get_feature(self): 
        print('get_feature ...')
        print('Encoding top ...')
        time.sleep(1)
        for img in tqdm(self.seg_top_db):
            img = cv2.resize(img, self.input_size)
            img = np.expand_dims(img, axis=0)
            img = self.preprocess_input(img)
            encoding = self.model_2.predict(img)[0]
            self.database_top_encodings.append(encoding)

        print('Encoding bottom ...')
        time.sleep(1)
        for img in tqdm(self.seg_bottom_db):
            img = cv2.resize(img, self.input_size)
            img = np.expand_dims(img, axis=0)
            img = self.preprocess_input(img)
            encoding = self.model_2.predict(img)[0]
            self.database_bottom_encodings.append(encoding)

        print('Encoding dress ...')
        time.sleep(1)
        for img in tqdm(self.seg_dress_db):
            img = cv2.resize(img, self.input_size)
            img = np.expand_dims(img, axis=0)
            img = self.preprocess_input(img)
            encoding = self.model_2.predict(img)[0]
            self.database_dress_encodings.append(encoding)
            
    def euclidean_distance(self, a, b):
        dist = np.linalg.norm(a - b)
        return dist
    
    def read_cate(self, img_path):
        name = img_path.split('/')[-1].split('.jpg')[0]
        cate = self.ec_data[self.ec_data["id"] == int(name.replace('p',''))]['category'].values[0]
        if cate == '上身類':
            return 'top'
        elif cate == '下身類':
            return 'bottom'
        elif cate == '洋裝/連身褲':
            return 'dress'
    
    def rec_clothing(self, img_path):
        cate_seg = {}
        test = cv2.imread(img_path)

        # Model 1
        test_result = inference_detector(self.model_1, test)

        # Draw bbox and segmentarion mask
        test_segmentation = self.show_result(test, test_result, self.model_1.CLASSES, score_thr=0.7)

        for cate, seg_item in zip(test_segmentation.keys(), test_segmentation.values()):
            if cate == 'top':
                seg_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                cate_seg['top'] = seg_item
            elif cate == 'skirt' or cate == 'leggings' or cate == 'pants':
                seg_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                cate_seg['bottom'] = seg_item
            elif cate == 'dress':
                seg_item = cv2.cvtColor(seg_item, cv2.COLOR_BGR2RGB)
                cate_seg['dress'] = seg_item

        # Model 2
        test = cv2.resize(seg_item, self.input_size)
        test = np.expand_dims(test, axis=0)
        test = self.preprocess_input(test)
        test_encoding = self.model_2.predict(test)[0]

        res_list=[]
        product_list=[]
        for cate, seg_item in zip(cate_seg.keys(), cate_seg.values()):   
            if cate == 'top':
                print("top")
                dis_list = []
                for i in self.database_top_encodings:
                    dis = self.euclidean_distance(test_encoding, i)
                    dis_list.append(dis)
                    dis_min = np.min(dis_list)
                for idx in list(pd.DataFrame(dis_list,columns=['dis']).sort_values('dis').head(5).index):
                    res_list.append(self.seg_top_db_file[idx])

            if cate == 'bottom':
                print("bottom")
                dis_list = []
                for i in self.database_bottom_encodings:
                    dis = self.euclidean_distance(test_encoding, i)
                    dis_list.append(dis)
                    dis_min = np.min(dis_list)  
                for idx in list(pd.DataFrame(dis_list,columns=['dis']).sort_values('dis').head(5).index):
                    res_list.append(self.seg_bottom_db_file[idx])

            if cate == 'dress':
                print("dress")
                dis_list = []
                for i in self.database_dress_encodings:
                    dis = self.euclidean_distance(test_encoding, i)
                    dis_list.append(dis)
                    dis_min = np.min(dis_list) 
                for idx in list(pd.DataFrame(dis_list,columns=['dis']).sort_values('dis').head(5).index):
                    res_list.append(self.seg_dress_db_file[idx])
            
        for item in res_list:
            _id=item.replace('../data/img/','').replace('.jpg','').replace('p','')
            product_list.append(self.ec_data[self.ec_data['id']==int(_id)][['title','price','special_price','img_url','url']].to_dict("record")[0])
            product_df=pd.DataFrame(product_list).fillna(0)
            product_df['price']=(product_df['price']+product_df['special_price']).astype('int')
            product_list=product_df[['title','price','img_url','url']].to_dict('recode')
        product_list
        return product_list
    
    def read_cate(self, img_path):
        name = img_path.split('/')[-1].split('.jpg')[0]
        cate = self.ec_data[self.ec_data["id"] == int(name.replace('p',''))]['category'].values[0]
        if cate == '上身類':
            return 'top'
        elif cate == '下身類':
            return 'bottom'
        elif cate == '洋裝/連身褲':
            return 'dress'

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), post_data.decode('utf-8'))
        if self.path=='/get_rec_product':
            self._set_response()
            # self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
            img_path=json.loads(post_data)['img_path'] 
            try:  
                res_list=simlar_clothing.rec_clothing(img_path)
                res=json.dumps({"status":True,"rusult":res_list})
                self.wfile.write(res.encode('utf-8'))
            except Exception as e:
                self._set_response()
                res=json.dumps({"status":False,"msg":str(e)})
                self.wfile.write(res.encode('utf-8'))
        else:
            self._set_response()
            self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

simlar_clothing=simlarClothing()
def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()


# def main(img_path):
#     simlar_clothing=simlarClothing()
#     res_list=simlar_clothing.rec_clothing(img_path)
#     return res_list

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("-img_path", default='./test2.jpg')
#     args = parser.parse_args(sys.argv[1:])
#     res_list=main(args.img_path)
#     print("result: ",res_list)
