## INSTALL ENV:
Install follow requirement.txt and install more 2 libs by command:
```
pip install openvino
pip install openvino-dev[extras]
```

## CONVERT MODEL YOLOV3 TINY OPENVINO:
```
#Download default model darknet

wget -O weights/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights

#Convert Darknet to keras

python tools/model_converter/convert.py cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights weights/yolov3-tiny.h5

#Convert Keras to TF2

python tools/model_converter/keras_to_tensorflow.py --input_model weights/yolov3-tiny.h5 --output_model weights/yolov3-tiny.pb

#convert TF2 or ONNX to OPENVINO
python convert_openvino.py --model_file weights/yolov3-tiny.pb --output_dir tweights
```
After converted model to openvino. You will see 2 files model openvino `yolov3-tiny.xml` and `yolov3-tiny.bin` in the `weights` directory.

## Run and compare results

Run default model keras (.h5) file:
```
python yolo.py --image
```

paste path to image and check result in output dir

Run openvino model:
```
python open_vino_run.py
```

paste path to image and check result in output dir