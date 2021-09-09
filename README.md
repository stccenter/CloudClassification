# **Introduction: Cloud Classification**

This project is implemented in two methods:
1. Standard CPU based - "cloud_combine.py" script is to train the DNN model for detecting cloudy pixels of Cross-track Infrared sounder using the top 75 Principal Components (PC)of it's spectrum.
2. GPU based


## **Software requirements**
1. Python 3.8
2. Python IDE

## **Standard CPU-based implementation**

#### **1. Clone the repository**

#### **2. Set up the virtual environment**
1. Create a new project folder and name it as per your wish. For example "cloudclassify".
2. Copy cloud_subpixel.py from cloned repository and place it inside cloudclassify folder.
3. In command prompt, go to cloudclassify folder and run the below command. This creates a virtual environment called "cloudclassify-venv"
   
            python -m venv cloudclassify-venv
4. In command prompt, run below line to activate the virtual environment
   
            cloudclassify-venv\Scripts\activate.bat
#### **3. Install python packages**
Copy requirements.txt file from cloned repository and place it inside cloudclassify folder. Use the following command to install all necessary packages at once:

            pip install -r requirements.txt     

#### **4. Download the data**
1. Click the [link](https://drive.google.com/drive/u/0/folders/1d9uS1EDtIkmTHu3pDhJgR7mbqVS2gZNM) and download the input data.
2. Place the downloaded "cloud_pc.sav" file inside cloudclassify folder

#### **5. Download train and test data**
   1. Download data using this <a href="https://drive.google.com/drive/folders/1XqrxJd6rGgd0N2QJXRd1fm5aCZiSZClR?usp=sharing" target="_blank">link</a>.
   2. Download all three folders to your local machine inside cloudclassify folder.
   
#### **6. Run the script**
Now, we are all set to run the script. 
Run the cloud_combine.py script using below command. This script accepts an argument called flag (-f). The default value of the flag is detection. 
1. In command prompt, run the script with default value "detection"

            python cloud_combine.py

2. In command prompt, run the script with flag "rainy cloud"

            python cloud_combine.py -f "rainy cloud"
         
#### **7. Output**
1. The model weights can be found in inside my_model folder.
2. The accuracy metrics (highlighted in yellow) such as probability of detection (POD), probability of false detection (POFD), false alarm ratio (FAR), bias, critical success index (CSI), and model accuracy and runtime (in seconds) will be printed in the terminal when the process finishes.
   | POD    | POFD   | FAR    | Bias   | CSI    | Accuracy |
   |--------|--------|--------|--------|--------|----------|
   | 0.6933 | 0.0431 | 0.1106 | 0.7795 | 0.6382 | 0.8690   |
3. The accuracy of the model is 86.90% and it took 91.69 minutes to finish. 
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/CPUOutput.png)

## **GPU-based implementation**
The GPU-based implementation is tested in three environments.
   1. Windows Desktop with NVIDIA GeForce GTX 1650 Ti with Max-Q Design
   2. AWS g4dn instance with NVIDIA Tesla T4
   3. Google Colab
   
## **1. Windows Desktop with NVIDIA GeForce GTX 1650 Ti with Max-Q Design**
### **Enable and install GPU driver**
#### **1. Verfiy graphic card details.**
   1. Go to Windows Start menu and type device manager. Expand Display Adapters, graphic cards will be displayed. 
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/DeviceManager.png)

#### **2. Download and install the NVIDIA driver**
   1. Go to NVIDIA drive download [link](https://www.nvidia.com/Download/index.aspx?lang=en-us)
   2. Provide NVIDIA driver details according to your NVIDIA product. Below screenshot shows the selection based on “NVIDIA GeForce GTX 1640 Ti with Max-Q Design”.
    ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/NVIDIADriver.png)
   3. Click on Search and download the driver. 
   4. Install downloaded NVIDIA driver. 
   5. You will find CUDA subfolder inside “NVIDIA GPU computing toolkit” folder inside C drive “Program Files” folder (C:\Program Files\NVIDIA GPU Computing Toolkit). 
   
#### **3. CUDA toolkit**
   1. Go to [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
   2. Find the latest release of CUDA Toolkit.
    ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/Cudatoolkit.png)
   3. Select the Operating System (Linux or Windows), architecture, and version based on your machine preference.
    ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/SelectTarget.png)
   4. Click download.
   5. Double click the downloaded exe file (Example: cuda_11.4.1_471.41_win10.exe) and follow the on-screen prompts.
   
#### **4. Download cuDNN library**
   1. Go to cuDNN [link](https://developer.nvidia.com/cudnn).
   2. Click Download cuDNN. If you are a first-time user, you need to create a user account and consent to the cuDNN Software License Agreement.
   3. Select the right version of cuDNN. Please note that the version of CUDA and cuDNN should match. In this case, we should download version 11.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/cudnn.png)
   4. It will download as a compressed folder. Extract the compressed folder.
   5. The extracted folder has “cuda” subfolder that matches with the “CUDA” subfolder in C:\Program Files\NVIDIA GPU Computing Toolkit.
   6. Now, copy cudnn64_8.dll from the bin of the extracted folder (C:\Users\anush\Downloads\cudnn-11.4-windows-x64-v8.2.2.26\cuda\bin) and paste it in the bin folder inside CUDA folder of NVIDIA GPU Computing Toolkit (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin).
   7. Copy cudnn.h file from include of the extracted folder (C:\Users\anush\Downloads\cudnn-11.4-windows-x64-v8.2.2.26\cuda\include) and paste it in the bin folder inside CUDA folder of NVIDIA_GPU_Computing Toolkit (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\include).
   8. Copy cudnn.lib file from lib/x64 folder inside extracted folder (C:\Users\anush\Downloads\cudnn-11.4-windows-x64-v8.2.2.26\cuda\lib\x64) and paste it in the similar folder of NVIDIA_GPU_Computing_Tookit (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64).
   
#### **5. Set up the virtual environment**
   1. Create a new project folder and name it as per your wish. For example "cloudclassify-gpu".
   2. Go to Start menu and type “command prompt”.
   3. Open command prompt. Change to your project folder.
   4. Copy and paste the below line in your command prompt. This creates a virtual environment named “cloudclassify-gpu”. You can name virtual environment as per your wish.

            python -m venv cloudclassify-gpu

   5. Copy and paste below line in command prompt. This activates the virtual environment.

            cloudclassify-gpu\Scripts\activate.bat

#### **6. Download train and test data**
   1. Download data using this <a href="https://drive.google.com/drive/folders/1XqrxJd6rGgd0N2QJXRd1fm5aCZiSZClR?usp=sharing" target="_blank">link</a>.
   2. Download all three folders to your local machine inside project folder.

#### **7. Install python packages**
   1. In command prompt, copy and paste below line to install python packages.
            
            pip install tensorflow-gpu==2.4.0 tensorboard==2.4.0 tensorboard-plugin-profile==2.4.0 scikit-learn pandas

#### **8. Verify the installation of GPU and run the script**
   1. In command prompt, run the python script using below command.
   
            python cloudcode.py

Note: Line #17 shows all the physical GPU devices available to TensorFlow. You should see device_type: “GPU” in the list of devices.

![image](https://github.com/stccenter/CloudClassification/blob/main/Images/verifygpu.png)


#### **9. Output of the script**
1. The model weights can be found inside my_model folder.
2. The accuracy metrics (highlighted in yellow) such as probability of detection (POD), probability of false detection (POFD), false alarm ratio (FAR), bias, critical success index (CSI), and model accuracy and runtime (in seconds) will be printed in the terminal when the process finishes.
   | POD    | POFD   | FAR    | Bias   | CSI    | Accuracy |
   |--------|--------|--------|--------|--------|----------|
   | 0.6789 | 0.0413 | 0.1086 | 0.7617 | 0.6271 | 0.8654   |
3. The accuracy of the model is 86.54% and it took 122.73 minutes to finish.
![image](https://github.com/stccenter/CloudClassification/blob/main/Images/output_gpu_laptop.png)

## **2. AWS g4dn instance with NVIDIA Tesla T4**
### **Install NVIDIA drivers on Windows instances**
Refer this [AWS help document](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/install-nvidia-driver.html#nvidia-gaming-driver) to follow the installation options.
The following steps are based on G4dn instance.
#### **1. Download and install the NVIDIA driver**
   1. Connect to the Windows instance.
   2. Go to Start menu and open a PowerShell window.
   3. Copy and paste the below Powershell command. This command will download driver inside Desktop.
   
               $Bucket = "nvidia-gaming"
               $KeyPrefix = "windows/latest"
               $LocalPath = "$home\Desktop\NVIDIA"
               $Objects = Get-S3Object -BucketName $Bucket -KeyPrefix $KeyPrefix -Region us-east-1
               foreach ($Object in $Objects) {
                  $LocalFileName = $Object.Key
                  if ($LocalFileName -ne '' -and $Object.Size -ne 0) {
                     $LocalFilePath = Join-Path $LocalPath $LocalFileName
                     Copy-S3Object -BucketName $Bucket -Key $Object.Key -LocalFile $LocalFilePath -Region us-east-1
                  }
               }

       ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/pscmd.PNG)

   4. Navigate to the desktop->NVIDIA  folder->windows->latest.
   5. Extract the zip folder.
   6. Double-click the installation file (exe) to launch it. 
   7. Follow the instructions to install the driver
   8. Reboot your instance as required. 
   9. To verify that the GPU is working properly, check Device Manager. Go to Start menu and open Device Manager. Expand Display adapters.
   
      ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/teslagpu.PNG)

   10. Create a registry value in the HKEY_LOCAL_MACHINE\SOFTWARE\NVIDIA Corporation\Global key with the name vGamingMarketplace, the type DWord, and the value 2.
       1. Use Powershell and run the below command.

               New-ItemProperty -Path "HKLM:\SOFTWARE\NVIDIA Corporation\Global" -Name "vGamingMarketplace" -PropertyType "DWord" -Value "2"

       2.  Use Powershell and run the below command.

               reg add "HKLM\SOFTWARE\NVIDIA Corporation\Global" /v vGamingMarketplace /t REG_DWORD /d 2

   11. Use Powershell and run the below command to download the certification file.
   
               Invoke-WebRequest -Uri "https://nvidia-gaming.s3.amazonaws.com/GridSwCert-Archive/GridSwCertWindows_2021_10_2.cert" -OutFile "$Env:PUBLIC\Documents\GridSwCert.txt"

   12. Reboot the instance.

#### **2. Install CUDA**
   1. Go to [NVIDIA website](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) and select the version of CUDA that you need.
   2. For **version**, choose based on your windows type and for **instance type**: choose exe (local)
   3. Click download.
   4. Double click the download exe file and follow the on-screen prompts.
   5. Reboot the instance. 

#### **3. Set up the virtual environment**
   1. Create a new project folder and name it as per your wish. For example "cloudclassifygpu".
   2. Go to Start menu and type “command prompt”.
   3. Open command prompt. Change to your project folder.
   4. Copy and paste the below line in your command prompt. This creates a virtual environment named “cloudclassify-gpu”. You can name virtual environment as per your wish.

            python -m venv cloudclassify-gpu

   5. Copy and paste below line in command prompt. This activates the virtual environment.

            cloudclassify-gpu\Scripts\activate.bat

#### **4. Install python packages**
   1. In command prompt, copy and paste below line to install python packages.
            
            pip install tensorflow-gpu==2.4.0 tensorboard==2.4.0 tensorboard-plugin-profile==2.4.0 scikit-learn pandas

#### **5. Download train and test data**
   1. Download data using this <a href="https://drive.google.com/drive/folders/1XqrxJd6rGgd0N2QJXRd1fm5aCZiSZClR?usp=sharing" target="_blank">link</a>.
   2. Download all three folders to your local machine inside project folder.
   
#### **6. Verify the installation of GPU and run the script**
   1. In command prompt, run the python script using below command.
   
            python cloudcode.py

Note: Line #17 shows all the physical GPU devices available to TensorFlow. You should see device_type: “GPU” in the list of devices.

![image](https://github.com/stccenter/CloudClassification/blob/main/Images/gpu.PNG)

#### **7. Output**
1. The model weights can be found inside my_model folder.
2. The accuracy metrics (highlighted in yellow) such as probability of detection (POD), probability of false detection (POFD), false alarm ratio (FAR), bias, critical success index (CSI), and model accuracy and runtime (in seconds) will be printed in the terminal when the process finishes.
   | POD    | POFD   | FAR    | Bias   | CSI    | Accuracy |
   |--------|--------|--------|--------|--------|----------|
   | 0.6816 | 0.0436 | 0.1134 | 0.7687 | 0.6269 | 0.8648   | 
3. The accuracy of the model is 86.48% and it took 243.54 minutes to finish.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/output_gpu_aws.PNG)


## **3. Open Google Colab**
1. Go to <a href="https://colab.research.google.com/notebooks/intro.ipynb#recent=true" target="_blank">Google Colab</a>.
2. Click on the Upload tab.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/GoogleColabUpload.png)
3. Choose the "CloudClassifyGPU.ipynb" downloaded from the Git repository.

#### **1. Download train and test data and upload them in your Google Drive**
1. Download data using this <a href="https://drive.google.com/drive/folders/1XqrxJd6rGgd0N2QJXRd1fm5aCZiSZClR?usp=sharing" target="_blank">link</a>.
2. Download all three folders to your local machine.
3. Then upload it to your Google drive.
4. Create a new folder called "logs" inside the same folder where you uploaded the above three folders.
5. Finally, your folder should have four folders. See screenshot below:
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/GDriveFolders.png)

#### **2. Change Runtime to GPU**
1. Go to Runtime in menu.
2. Click "Change runtime type".
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/ChangeRunTime.png)
3. In Notebook setting, change Hardware accelerator to "GPU" and click Save.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/RunTimeGPU.png)
   

#### **3. Run the notebook**
1. Run each code block sequentially.
2. Note when you run the "GPU available" block, you should see below if GPU is properly enabled.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/GPUEnabled.jpg)
   Note: If you encounter Import error (screenshot below): cannot import name LayerNormalization, go to Runtime -> Factory Reset runtime. Then set "Hardware Accelerator" to GPU and save it.
   ![image](https://github.com/stccenter/CloudClassification/blob/main/Images/Error.PNG)

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

#### **4. Output**
1. The model weights can be found inside the my_model folder.
2. The accuracy metrics (highlighted in yellow) such as probability of detection (POD), probability of false detection (POFD), false alarm ratio (FAR), bias, critical success index (CSI), and model accuracy and runtime (in seconds) will be printed in the code block.
   | POD    | POFD   | FAR    | Bias   | CSI    | Accuracy |
   |--------|--------|--------|--------|--------|----------|
   | 0.6903 | 0.0440 | 0.1131 | 0.7783 | 0.6344 | 0.8674   |
3. The accuracy of the model is 86.74% and it took 91.62 minutes to finish.

## **Walkthrough Video**
**CPU-based implementation**\
[<img src="https://github.com/stccenter/CloudClassification/blob/main/Images/Videos.jpg" width="60%">](https://youtu.be/6mvsTfZtE-M)\

**GPU-based implementation**\
[<img src="https://github.com/stccenter/CloudClassification/blob/main/Images/RunTimeGPU.png" width="60%">](https://youtu.be/Pzlsyb4s5yQ)


