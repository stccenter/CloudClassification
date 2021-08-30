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
Use the following command to install all necessary packages at once:

            pip install -r requirements.txt     

#### **IV Download the data**
1. Click the [link](https://drive.google.com/drive/u/0/folders/1d9uS1EDtIkmTHu3pDhJgR7mbqVS2gZNM) and download the input data.
2. Place the downloaded "cloud_pc.sav" file inside cloudclassify folder

#### **V - Run the script**
Now, we are all set to run the script. 
Run the cloud_combine.py script using below command. This script accepts an argument called flag (-f). The default value of the flag is detection. 
1. Run the script with default value "detection"

            python cloud_combine.py

2. Run the script with flag "rainy cloud"

            python cloud_combine.py -f "rainy cloud"
         
#### **VI - Output**
1. Once the script is finished running, you will find the output inside my_model folder.

### **GPU-based implementation**

#### **I - Open Google Colab**
1. Go to <a href="https://colab.research.google.com/notebooks/intro.ipynb#recent=true" target="_blank">Google Colab</a>.
2. Click on the Upload tab.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/GoogleColabUpload.png)
3. Choose the "CloudClassifyGPU.ipynb" downloaded from the Git repository.

#### **II - Download train and test data and upload them in your Google Drive**
1. Download data using this <a href="https://drive.google.com/drive/folders/1XqrxJd6rGgd0N2QJXRd1fm5aCZiSZClR?usp=sharing" target="_blank">link</a>.
2. Download all three folders to your local machine.
3. Then upload it to your Google drive.
4. Create a new folder called "logs" inside the same folder where you uploaded the above three folders.
5. Finally, your folder should have four folders. See screenshot below:
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/GDriveFolders.png)

#### **III - Change Runtime to GPU**
1. Go to Runtime in menu.
2. Click "Change runtime type".
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/ChangeRunTime.png)
3. In Notebook setting, change Hardware accelerator to "GPU" and click Save.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/RunTimeGPU.png)
   

#### **IV - Run the notebook**
1. Run each code block sequentially.
2. Note when you run the "GPU available" block, you should see below if GPU is properly enabled.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/GPUEnabled.jpg)
3. When you run the code block "Mount the Google drive".
   1. Click on the URL.
    ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/MountGdrive.PNG)
   2. Obtain the authorization code and paste it in the "Enter your authorization code" box.
   3. Verify Google drive is mounted correctly.
      1. Click file icon in the left panel.
      2. Click Mount Drive (third option).
      3. Click "Connect to Google Drive" when it asks for permission. It takes few minutes to mount.
      4. Once it is mounted, you should see drive folder. See screenshot below:
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/VerifyGdrive.png)
4. Please make sure to check your data path if you get an IOError in the below code block.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/LoadDataPath.png)


### **Videos**
**CPU-based implementation**
[<img src="https://github.com/stccenter/CloudClassification/blob/main/Images/Videos.jpg" width="60%">](https://youtu.be/vFepWVEbl0I)

