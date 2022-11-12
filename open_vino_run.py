from openvino.runtime import Core
import numpy as np
import time
import os
from pathlib import Path
from PIL import Image
from yolo3.postprocess_np import yolo3_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes, optimize_tf_gpu


class YOLO_OpenVino(object):
    def __init__(self, path_xml, model_input_shape, classes_path, anchors_path):
        self.path_xml = path_xml
        self.model_input_shape = model_input_shape
        self.class_names = get_classes(classes_path)
        self.anchors = get_anchors(anchors_path)
        self.colors = get_colors(len(self.class_names))
        self.score = 0.1
        self.iou = 0.4
        self.elim_grid_sense = False
        self._generate_model()

    def _generate_model(self):
        ie = Core()
        if not Path(self.path_xml).is_file():  # if not *.xml
            w = next(Path(self.path_xml).glob('*.xml'))  # get *.xml file from *_openvino_model dir
        network = ie.read_model(model=self.path_xml, weights=Path(self.path_xml).with_suffix('.bin'))
        self.inference_model = ie.compile_model(model=network, device_name="CPU")

    def detect_image(self, image):
        if self.model_input_shape != (None, None):
            assert self.model_input_shape[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_input_shape[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_input_shape)
        #origin image shape, in (height, width) format
        image_shape = image.size[::-1]

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores


    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        pred = self.inference_model([image_data])
        preds = []
        for k in pred.keys():
            preds.append(pred[k])
        out_boxes, out_classes, out_scores = yolo3_postprocess_np(preds,
                                                                  image_shape,
                                                                  self.anchors,
                                                                  len(self.class_names),
                                                                  self.model_input_shape,
                                                                  max_boxes=100,
                                                                  confidence=self.score,
                                                                  iou_threshold=self.iou,
                                                                  elim_grid_sense=self.elim_grid_sense)

        return out_boxes, out_classes, out_scores


    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)
        
if __name__ == '__main__':
    yolov_openvino = YOLO_OpenVino(path_xml="weights/yolov3-tiny.xml",
                                   model_input_shape=(416, 416),
                                   classes_path="configs/coco_classes.txt",
                                   anchors_path="configs/tiny_yolo3_anchors.txt")
    
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img).convert('RGB')
        except:
            print('Open Error! Try again!')
            continue
        else:
            base_name = os.path.basename(img)
            r_image, _, _, _ = yolov_openvino.detect_image(image)
            r_image.show()
            path_save = os.path.join("output", base_name+'_openvino.jpg')
            r_image.save(path_save)