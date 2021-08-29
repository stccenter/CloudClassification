### **Introduction**
This git repository is created for cloud classification project. This project is implemented in two methods:
1. Standard CPU based - "cloud_combine.py" script is to train the DNN model for detecting cloudy pixels of Cross-track Infrared sounder using the top 75 Principal Components (PC)of it's spectrum.
2. GPU based


### **Software requirements**
1. Python 3.7 or above
2. Python IDE (Visual Studio Code)

### **Standard CPU-based implementation**

#### **I - Clone the repository**

#### **II - Set up the virtual environment**
1. Create a new folder and name it as cloudclassify.
2. Copy cloud_subpixel.py from cloned repository and place it inside cloudclassify folder.
3. In Visual Studio Code, go to Terminal and run the below in cmd terminal. This creates a virtual environment called "cloudclassify-venv"
   
            python -m venv cloudclassify-venv
4. For Windows, run below line to activate the virtual environment
   
            cloudclassify-venv\Scripts\activate.bat
#### **III - Install python packages**
      
            pip install TensorFlow scipy scikit-learn matplotlib       

#### **IV Download the data**
1. Click the [link](https://drive.google.com/drive/u/0/folders/1d9uS1EDtIkmTHu3pDhJgR7mbqVS2gZNM) and download the input data.
2. Place the downloaded "cloud_pc.sav" file inside cloudclassify folder

#### **V - Run the script**
Now, we are all set to run the script.

### **GPU-based implementation**

#### **I - Open Google Colab**
1. Go to the [link](https://colab.research.google.com/notebooks/intro.ipynb#recent=true).
2. Click on the Upload tab.
   ![image](https://user-images.githubusercontent.com/86536464/125887893-a205773b-67d4-4d0d-b87a-5dc1e567bc99.png)

#### **II - Download the train and test data and upload in GDrive**

#### **III - Mount the GDrive**

#### **IV - Run the notebook**
