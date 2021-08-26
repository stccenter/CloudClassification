### **Introduction**
This git repository is created for cloud classification project. "cloud_subpixel" script is to train the DNN model for detecting cloudy pixels of Cross-track Infrared sounder using the top 75 Principal Components (PC)of it's spectrum 

### **Software requirements**
1. Python 3.7 or above
2. Python IDE (Visual Studio Code)

### **Steps**

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
