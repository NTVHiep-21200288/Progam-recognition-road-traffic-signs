# Traffic Sign Classification

This project is a traffic sign classification system that uses a convolutional neural network (CNN) to classify traffic signs. The project consists of two main components:

1. **Training the model** (`trafficsign.py`)
2. **Predicting traffic signs** (`predict.py`)

## Requirements

The project requires the following Python libraries:

- `cv2` (OpenCV)
- `numpy`
- `tensorflow`
- `matplotlib`
- `scikit-learn`

These can be installed using the `requirements.txt` file provided.

## Dataset

The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB). You can download it from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Once downloaded, extract the dataset and organize it
## Training the Model

Run the following command to start training:

```bash
python trafficsign.py gtsrb [model.h5]
```
- gtsrb: Path to the training images directory.
- model.h5 (optional): Path to save the trained model.
## Predicting Traffic Signs

To predict traffic signs using a trained model, run the following command:
    
```bash
python predict.py model.h5
```
- model.h5: Path to the trained model file.  

You will be prompted to enter the path to an image file. The program will then display the predicted traffic sign category and its confidence.

```bash
Enter the path to the image file (or 'exit' to quit): cong_truong.png
```