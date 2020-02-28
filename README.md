# a. Follow this guide for installing CUDA and CUDNN:
https://github.com/markjay4k/How-To-Install-TensorFlow-GPU/blob/master/How%20To%20Install%20TensorFlow%201.4.ipynb
# b. Download TensorFlow Object Detection API repository from GitHub
https://github.com/tensorflow/models
# c. Download a specific model
# d. Download this tutorial's repository from GitHub
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

# 1. If you want to train your own object detector, delete the following files (do not delete the folders):
All files in \object_detection\images\train and \object_detection\images\test
The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
All files in \object_detection\training
All files in \object_detection\inference_graph

# 2. Anaconda
```
conda create -n tensorflow1 pip python=3.7
activate tensorflow1
python -m pip install --upgrade pip
pip install --ignore-installed --upgrade tensorflow-gpu
```
# 3. Install Packages
```
conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install Cython
pip install contextlib2
pip install jupyter
pip install matplotlib
pip install pandas
pip install opencv-python
```
# 4. Set Path
```
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
# 5. Compile Protobufs and run setup.py
```
cd /d C:\tensorflow1\models\research
```
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
```
python setup.py build
python setup.py install
```
# 6. Test TensorFlow setup to verify it works
```
cd object_detection
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
jupyter notebook object_detection_tutorial.ipynb
```
Gather and Label Pictures
https://github.com/tzutalin/labelImg
```
cd C:\labelImg
conda install pyqt=5
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```
# 7. Generate Training Data
```
python xml_to_csv.py
```
Open the generate_tfrecord.py file in a text editor and
replace the label map starting at line 31 with your own label map
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
# 8. Create Label Map and Configure Training
Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder.
```
item {
  id: 1
  name: 'nine'
}

item {
  id: 2
  name: 'ten'
}
```
# 9. Configure training
Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the ssd_mobilenet_v2_quantized_300x300_coco.config file into the \object_detection\training directory. 
Then, open the file with a text editor.

- Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 3 .
- Line 106. Change fine_tune_checkpoint to:
fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt"
- Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:
input_path : "C:/tensorflow1/models/research/object_detection/train.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
- Line 130. Change num_examples to the number of images you have in the \images\test directory.
- Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:
input_path : "C:/tensorflow1/models/research/object_detection/test.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

# 10. Run the Training
Simply move train.py from /object_detection/legacy into the /object_detection folder
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```