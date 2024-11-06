# SVM Classifier for Cancer Patient Data
This Python script applies Support Vector Machine (SVM) classifiers with different kernels on a dataset of cancer patients to predict the severity of the condition based on two factors: Air Pollution and Genetic Risk. The dataset is used to train and evaluate the performance of SVM models with different kernel functions (linear, polynomial, and radial basis function - RBF).

## Requirements
- **Libraries:**
- **Python 3.x**
- **numpy**
- **matplotlib**
- **sklearn**
- **pandas**


To install the required libraries, run:

```bash
pip install numpy matplotlib scikit-learn pandas
```
### Dataset
The dataset used in this script is assumed to be a CSV file. The dataset includes the following columns:

```Air Pollution```: A measure of the air pollution in the area of the patient.

```Genetic Risk```: A measure of the genetic risk factor.

```Level```: The severity of the cancer, categorized as Low, Medium, and High.

The ```Level``` column is mapped to integer values as follows:

```Low``` -> 0

```Medium``` -> 0

```High``` -> 2

## Code Overview
### Data Preparation
Loading Data: The dataset is read into a ```pandas``` DataFrame.

Data Preprocessing: The Level column is mapped to integer values to represent different severity levels. A subset of the dataset is then selected with ```Air Pollution``` and ```Genetic Risk``` as features (X_data) and ```Level``` as the target variable (y_data).

Train-Test Split: The dataset is split into training and test sets (70% training, 30% testing).
### SVM Classifier
The script applies the following SVM models:

- **Linear Kernel: SVM model with a linear kernel.**
- **Polynomial Kernel: SVM model with a polynomial kernel.**
- **Radial Basis Function (RBF) Kernel: SVM model with the RBF kernel.**
- **Model Training and Evaluation:** For each model, the classifier is trained on the training set and then used to predict the Level on the test set.
The accuracy of the predictions is calculated using the ```accuracy_score``` function. The decision boundaries for each model are visualized using ```matplotlib```, showing the classification regions for each kernel.

### Accuracy Results
The accuracy of each model is printed to the console as a percentage.

### Decision Boundaries
For each model, a contour plot is displayed that visualizes the decision boundaries of the classifier. The plot uses the two features ```Air Pollution``` and ```Genetic Risk``` and colors the regions based on the predicted class labels. The training data points are plotted as well, with colors corresponding to their actual classes.


Link to Colab and view results:
```https://colab.research.google.com/gist/VajiheMobasheri/021284283281c655ae0e225ea664fd3f/main.ipynb```
