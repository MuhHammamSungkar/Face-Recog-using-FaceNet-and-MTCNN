# Face Detect using MTCNN and Recognition using FaceNet

## Brief Information:
My final version program is the `facenetv2.ipynb`

To run my system, the steps are:
  1. Open your favorite code editor
  2. Run `facenetv2.ipynb`

### Explanation
My system is recognizing someone face in the dataset. If the system detected an unregistered face at the camera, the system will label the face as **'Unknown'**. If the user wants to register, the system will ask their name and the camera will start capturing their face then train their captured faces and recognize them.

## Face Detection
MTCNN (Multitask Cascaded Convolutional Networks) is a face detection algorithm consisting of three convolutional networks that work sequentially: face detection network, face adjustment network, and face point determination network.
