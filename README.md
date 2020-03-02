# a. Follow this guide for installing Anaconda, CUDA and cuDNN:
I downloaded CUDA v10.1 and cuDNN v7.6, but check the table for compatibility:

table: https://www.tensorflow.org/install/source#tested_build_configurations

https://github.com/markjay4k/How-To-Install-TensorFlow-GPU/blob/master/How%20To%20Install%20TensorFlow%201.4.ipynb
# b. Download TensorFlow Object Detection API repository from GitHub
Create a folder in C:/ and name it tensorflow1 and move the download inside the folder. Rename the folder from "model-master" to "models".

https://github.com/tensorflow/models
# c. Download a specific model from Google's Model Zoo
I used the ssd_mobilenet_v2_quantized_coco model

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# d. Download this tutorial's repository from GitHub
Place the contents inside C:/tensorflow1/models/research/object_detection.

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

# e. Add paths to Environment Variables
Change directory to correct drive if necessary
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\libx64

# f. Install Visual C++ Build Tools 2015
https://visualstudio.microsoft.com/vs/older-downloads/

Download and install the following two packages:
- Microsoft Build Tools 2015 Update 3
- Microsoft Visual C++ 2015 Redistributable Update 3

Microsoft Visual Studio Code:

https://code.visualstudio.com/

Windows SDK 10:

https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/

Restart your computer


# 1. If you want to train your own object detector, delete the following files (do not delete the folders):
- All files in \object_detection\images\train and \object_detection\images\test
- The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
- All files in \object_detection\training
- All files in \object_detection\inference_graph

# 2. Set up new Anaconda virtual environment
Run Anaconda as administrator and run these commands:
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
conda install -c anaconda git
```
# 4. Configure PYTHONPATH environment variable
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
Go into the object detection folder and install pycocotools before running jupyter notebook. Run through the script to see if the object detection is working.
```
cd object_detection
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
jupyter notebook object_detection_tutorial.ipynb
```
# 7. Gather and Label Pictures (taken from link):

https://github.com/tzutalin/labelImg
```
cd C:\labelImg
conda install pyqt=5
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```
# 8. Generate Training Data
```
python xml_to_csv.py
```
Open the generate_tfrecord.py file in a text editor and
replace the label map starting at line 31 with your own label map.
On line 23, change the line to 'import tensorflow.compat.v1 as tf'.
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
# 9. Create Label Map and Configure Training
Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder.
```
item {
  id: 1
  name: 'bird'
}
```
# 10. Configure training
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

# 11. Run the Training
Simply move train.py from /object_detection/legacy into the /object_detection folder.
I had to downgrade to TensorFlow version 1.15 because of compatibility issues with the code.
```
pip install --ignore-installed --upgrade tensorflow-gpu==1.15.0
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config
```
Run until the loss is consistently under 2.

# 12. Export Inference Graph for TFLite
issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:
```
mkdir TFLite_model
set CONFIG_FILE=C:\\tensorflow1\models\research\object_detection\training\ssd_mobilenet_v2_quantized_300x300_coco.config
set CHECKPOINT_PATH=C:\\tensorflow1\models\research\object_detection\training\model.ckpt-XXXX
set OUTPUT_DIR=C:\\tensorflow1\models\research\object_detection\TFLite_model
python export_tflite_ssd_graph.py --pipeline_config_path=%CONFIG_FILE% --trained_checkpoint_prefix=%CHECKPOINT_PATH% --output_directory=%OUTPUT_DIR% --add_postprocessing_op=true
```

# 13. Install MSYS2 
https://www.msys2.org/

Open MSYS2 and run the commands:
```
pacman -Syu
```
Close the window, re-open it and issue the commands:
```
pacman -Su
pacman -S patch unzip
```
# 14. Update Anaconda and create tensorflow-build environment
In Anaconda:
```
conda update -n base -c defaults conda
conda update --all
#conda install -c anaconda vs2015_runtime
conda create -n tensorflow-build pip python=3.6
conda activate tensorflow-build
```
Then run these commands: (change PATH depending on which directory msys64 is located in)
```
python -m pip install --upgrade pip
set PATH=%PATH%;E:\msys64\usr\bin
```

# 15. Download Bazel and Python package dependencies
Change 'conda install -c conda-forge bazel=0.24.1' to the required version of bazel
```
pip install six numpy wheel
pip install keras_applications==1.0.6 --no-deps
pip install keras_preprocessing==1.0.5 --no-deps
#conda install -c anaconda openjdk
#conda install -c anaconda vs2013_runtime
conda install -c conda-forge bazel=0.24.1
```
Min: 24.1; Max: 26.1
# 16. Download TensorFlow source and configure build
Change 'git checkout r1.15' to the same version of TensorFlow used for training
```
cd /d C:\
mkdir C:\tensorflow-build
cd C:\tensorflow-build
git clone https://github.com/tensorflow/tensorflow.git 
cd tensorflow 
git checkout r1.15
python ./configure.py
```
During the prompts, enter:
- Enter
- Enter
- N
- N
- N
- Enter
- N

# 17. Build TensorFlow package
First go into the WORSPACE file in the C:/tensorflow-build/tensorflow directory and add the following to the top:
```
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "dc97fccceacd4c6be14e800b2a00693d5e8d07f69ee187babfd04a80a9f8e250",
    strip_prefix = "rules_docker-0.14.1",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.14.1/rules_docker-v0.14.1.tar.gz"],
)
```
Then run the commands:
```
try: bazel build --config=v1 //tensorflow/tools/pip_package:build_pip_package
add: --define=no_tensorflow_py_deps=true (to avoid issue with package creation)
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package 
bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg 
```
Error (1): "stderr /usr/bin/bash patch command not found"

Fix (1): start from step 13

Error (2): "fatal error C1189: #error: LLVM requires at least MSVC 2017"

Fix (2): 
# 18. Install TensorFlow and test it out!
TensorFlow is finally ready to be installed! Open File Explorer and browse to the C:\tmp\tensorflow_pkg folder. Copy the full filename of the .whl file, and paste it in the following command:
```
pip3 install C:/tmp/tensorflow_pkg/<Paste full .whl filename here>
python
import tensorflow as tf
tf.__version__
exit()
```

# 19. Use TOCO to Create Optimzed TensorFlow Lite Model, Create Label Map, Run Model
Create optimized TensorFlow Lite model
```
activate tensorflow-build
cd C:\tensorflow-build
set OUTPUT_DIR=C:\\tensorflow1\models\research\object_detection\TFLite_model
```
```
bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=%OUTPUT_DIR%/tflite_graph.pb --output_file=%OUTPUT_DIR%/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops 
```
```
bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=$OUTPUT_DIR/tflite_graph.pb --output_file=$OUTPUT_DIR/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --inference_type=FLOAT --allow_custom_ops 
```
