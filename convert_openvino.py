#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO keras model to ONNX model
"""
import os, sys, argparse
import shutil
import subprocess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects

def export_openvino(path_file, path_save):
    # YOLOv5 OpenVINO export'
    cmd = "mo --input_model {} --output_dir {} ".format(path_file, path_save)
    subprocess.run(cmd.split(), check=True, env=os.environ)  # export


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert YOLO from tensorflow2 or onnx to openvino model')
    parser.add_argument('--model_file', required=True, type=str, help='path to onnx, tensorflow2 (.onnx or .pb) model file')
    parser.add_argument('--output_dir', required=True, type=str, help='output onnx model file')


    args = parser.parse_args()
    export_openvino(args.model_file, args.output_dir)


if __name__ == '__main__':
    #requirement 
    '''
    
    '''
    main()

