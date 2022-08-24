# Neuroscope: An Explainable AI Toolbox for Semantic Segmentation and Image Classification of Convolutional Neural Nets
Trust in artificial intelligence (AI) predictions is a crucial point for a widespread acceptance of new technologies,
especially in sensitive areas like autonomous driving. The need for tools explaining AI for deep learning of images is
thus eminent. Our proposed toolbox Neuroscope addresses this demand by offering state-of-the-art visualization algorithms
for image classification and newly adapted methods for semantic segmentation of convolutional neural nets (CNNs). With
its easy to use graphical user interface (GUI), it provides visualization on all layers of a CNN. Due to its open
model-view-controller architecture, networks generated and trained with Keras and PyTorch are processable, with an
interface allowing extension to additional frameworks. We demonstrate the explanation abilities provided by Neuroscope
using the example of traffic scene analysis.
## Installation

### Setting up the environment

1. Install the latest version of [conda](https://www.anaconda.com/).
2. You can choose to install a CPU or GPU version. For GPU, you first need to install [cuda](https://developer.nvidia.com/cuda-toolkit) and [cudnn](https://developer.nvidia.com/cudnn).
3. In our installation, we will create a conda environment from a predefined file.  
If you're in the `neuroscope/` folder, the path for the CPU version would be `conda_environment/neuroscope_cpu.yml`.  
The GPU version, `neuroscope_gpu.yml`, is located in the same directory.  
To install an environment, open Anaconda Prompt and run `conda env create -f %path_to_the_yml_file%`.  
4. To verify that the environment was installed correctly, run `conda env list`. You should see a new environment on the list.

### Setting up PyCharm configuration

The next steps are provided for PyCharm, but you can use them as a reference for a different IDE.

1. In `File > Settings > Project:neuroscope > Project Interpreter`, click on the cog on the right.  
Select `Add > Conda environment > Existing environment`, and specify `python.exe` from the environment that was created in the previous steps.  
After selecting the interpreter, you should see many packages identified from the conda environment.  
2. In `Run > Edit Configurations`, in the upper-left corner, click `+` and select `Python`.  
In the opened window, select `Python Interpreter` if it's not autoselected.  
In the `Script path` field, specify a full path to `src/main.py`.  
In the `Working directory` field, specify a full path to the `neuroscope/` folder.  
4. To test that everything works as intended, run the `main.py` file with the configuration.

## Self-tests

### Technical information about testing

The test scripts are located in the `test\` directory.  
The tests use [unittest](https://docs.python.org/3/library/unittest.html) and [QTest](https://doc.qt.io/qt-5/qttest-index.html) libraries.  
The file `test_neuroscope_gui.py` contains test cases and `test_setting.py` contains fixed values related to the test data like properties and file paths.  
The `unittest` framework is used for starting and finalizing test cases with methods `setUp`/`tearDown` and the class `setUpClass`.  
Every test-case starts with the `test_` keyword.

### Running test scripts

To run the tests with PyCharm, complete the following steps:

1. In `Run > Edit Configurations`, click `+` and select `Python test > Unittests`. A new configuration will be created.  
2. In the new configuration, below `Target`, click on the folder icon and specify the full path to `neurscope\test\neuroscope_gui.py`.  
3. In the `Environment` section, specify the preferred `Python interpreter`.  
It's recommended to use the project default interpreter that was selected in `File > Settings > Project Interpreter`.  
4. Set a full path to `neurscope\test` in the `Working directory` field.  
5. In the upper-right side of the main PyCharm window, on the left to the `Run` and `Debug` buttons, there is a menu where you can select a configuration.  
Select the configuration that you have created in previous steps.  
6. To run the tests, `Run` or `Debug` the `test_neuroscope_gui.py` file.

If you want to run tests with a terminal:

1. Add `\neuroscope\test` to `PYTHONPATH`, i.e. `set PYTHONPATH=%PYTHONPATH%;C:\full_path_to_the_test_folder`.
2. In the terminal, go to the `\neuroscope\test` and run `python test_neuroscope_gui.py`.

## Usage

To launch Neuroscope in a terminal, go to the `neuroscope` folder and run `python src/main.py`.  
To launch Neuroscope in PyCharm, run `main.py` with a configuration that was created in the Installation step of this manual.

The common usage scenario consists of opening the model, adjusting the settings for the model, selecting images, applying analysis methods and saving the results.  
Here's how to do it:

* In the upper-left side of a Neuroscope window, click on the leftmost icon to select a model.
* After the model is selected, you will see its scheme on the left. Above it will be the icon with two cogs where you will be able to change the following settings for the model:
  * Contexts - This setting depends on the model. Choose what it was trained for.
  * preprocessing_presets - The preprocessing presets are located in `data\preprocessing_presets.json`. Mean and Standard Deviation fields are automatically filled from the preset.
  * If you want to create your own preprocessing preset, you can add a new one with the `new` button.
  * Normalization - When ticked, a selected preset is applied. When empty, the preset is not applied.
  * channel_first - This setting is about input images. To decide on this setting, you need to know how the format of your data handles channels, that is if the number of channels goes first or not. If it does, tick this field.
  * Image is RGB - When ticked, assumes that the image is given as RGB. When empty, assumes BGR.
  * Decoding File Path - The model usually produces a one-dimensional array of numbers. The `decoding` is about how to interpret this array to make it readable for a human.
  * Output Dimension - To enter the values in this field, you need to know what network gives as an output.
  * Output WHC - Similar to the `channel_first`, but for the output. WHC is Width Height Channels. When ticked, assumes that they go in this order. When empty assumes the reversed order.
  * Output activation - Some models don't have the softmax layer at the end of the network. If it so, you can tick this option, and Neuroscope will automatically add the softmax layer at the end.
* After adjusting model settings, open the images by pressing the second-to-left icon on top. If the window with the pictures didn't open, try `Window > New inspection window`. If the window still didn't open, there might be something wrong with the model settings. Some of the options are described below:
  * Prediction - By pressing this button, you process the images through the network. The Neuroscope doesn't automatically feed images to the network, so you don't lose results if you select another model.
  * Filter - postprocessing 
Setting - it's for the current inspection method, or related to the algorithm.
    
## Cite
```
@Article{app11052199,
AUTHOR = {Schorr, Christian and Goodarzi, Payman and Chen, Fei and Dahmen, Tim},
TITLE = {Neuroscope: An Explainable AI Toolbox for Semantic Segmentation and Image Classification of Convolutional Neural Nets},
JOURNAL = {Applied Sciences},
VOLUME = {11},
YEAR = {2021},
NUMBER = {5},
ARTICLE-NUMBER = {2199},
URL = {https://www.mdpi.com/2076-3417/11/5/2199},
ISSN = {2076-3417},
ABSTRACT = {Trust in artificial intelligence (AI) predictions is a crucial point for a widespread acceptance of new technologies, especially in sensitive areas like autonomous driving. The need for tools explaining AI for deep learning of images is thus eminent. Our proposed toolbox Neuroscope addresses this demand by offering state-of-the-art visualization algorithms for image classification and newly adapted methods for semantic segmentation of convolutional neural nets (CNNs). With its easy to use graphical user interface (GUI), it provides visualization on all layers of a CNN. Due to its open model-view-controller architecture, networks generated and trained with Keras and PyTorch are processable, with an interface allowing extension to additional frameworks. We demonstrate the explanation abilities provided by Neuroscope using the example of traffic scene analysis.},
DOI = {10.3390/app11052199}
}
```
