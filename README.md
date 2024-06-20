# Parking Occupancy Detection

Group project for the Computer Vision '23-'24 course, Sapienza University of Rome.

## Members

| **Name and Surname** |                                                                                      **GitHub**                                                                                      |
| :------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   `Dariana Dorin`    |     <a href="https://github.com/darianado"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c2/GitHub_Invertocat_Logo.svg" align="center" height="70" width="70" ></a>     |
| `Djibril Coulybaly`  | <a href="https://github.com/djibril-coulybaly"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c2/GitHub_Invertocat_Logo.svg" align="center" height="70" width="70" ></a> |

## Table of Contents

1. [About](#about)
2. [Components](#components)
3. [Models Used](#models-used)
4. [How to run](#how-to-run)

## About <a name="about"></a>

This project aims to detect whether a parking space is occupied by a vehicle/object or not using computer vision. By segmenting parking spaces from video data using a mask and trained models, we can perform real-time monitoring/data on occupancy count. This project can be applied to parking spaces in shopping centres, garages, airports etc.

## Componets <a name="components"></a>

1. 🎬 **Video Feed**
2. 📷 **Mask from Video Feed**
3. 📁 **Dataset**
4. 💻 **Technologies**

   - **OpenCV**: Image/Video Processing
   - **TensorFlow**: Building, training and testing deep learning models
   - **MatPlotLib**: Visual Representation of data
   - **SKLearn**: Evaluating model performance
   - **Python**: Programming language to complete project
   - **Numpy**: Handling numerical operations

## Models Used <a name="models-used"></a>

The following models were implemented and evaluated as part of a top-down approach:

1. **MobileNetV2**
2. **Keras Sequential Model**
3. **Random Forest**
4. **SVC**

## How To Run <a name="how-to-run"></a>

#### 1. Create Mask

Create the mask by executing the following command:

```python
python create_mask.py
```

This will produce an image of the video frame from which you can manually draw the bounding box of a fixed size by clicking the area you wish to draw. Once complete, press the ESC key to complete and save the mask

#### 2. Apply Mask

Apply the mask by executing the following command:

```python
python apply_mask.py
```

A window will appear with the mask successfully applied to the video feed

#### 3. Create, train and Save models

Create, train and save the models by executing the following command:

- **MobileNetV2**

In the MobileNetV2 folder:
```python
python save_model.py
```

- **Keras Sequential Model**

In the Sequential folder:
```python
python save_model.py
```

- **Random Forest**

In the HOG folder:

```python
python save_model_rf.py
```

- **SVC**

In the HOG folder:
```python
python save_model_svc.py
```

#### 3. Predict models on video feed

Predict the models on the video feed by executing the following command:

- **MobileNetV2**

In the MobileNetV2 folder:
```python
python predict.py
```

- **Keras Sequential Model**

In the Sequential folder:
```python
python predict.py
```

- **Random Forest**

In HOG folder modify first line in predict.py with
```python
model_filename = "random_forest_model.pkl"
```

Then in terminal run
```python
python predict.py
```

- **SVC**

In HOG folder modify first line in predict.py with
```python
model_filename = "svc_model.pkl"
```

Then in terminal run
```python
python predict.py
```
