import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageTk, Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#loading the csv data to a pandas Data frame
df=pd.read_csv('FinalDengueData.csv')
df.drop('Gender', axis = 1, inplace = True)
df.drop('Id',axis=1,inplace = True)
X=df.drop(columns='Target',axis=1)
Y=df['Target']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=3)
model=LogisticRegression()
model.fit(X_train,Y_train)
clfd = DecisionTreeClassifier(criterion ="entropy", max_depth = 4)
clfd.fit(X_train,Y_train.ravel())
y_test_pred = clfd.predict(X_train)

def add_inputs():
    inputs = []
    for entry in entries:
        inputs.append(float(entry.get()))
    tuple_input = tuple(inputs)
    array_input_data = np.asarray(tuple_input)
    #reshape the numpy array (predicting on only on instance)
    input_data_reshape = array_input_data.reshape(1,-1)
    #Prediction
    prediction = model.predict(input_data_reshape)
    if(prediction==1):
        result_label.configure(text=f"You are Normal")
        print("Normal")
    elif(prediction==2):
        result_label.configure(text=f"The result is Low")
        print("Low")
    elif(prediction==3):
        result_label.configure(text=f"The result is High")
        print("High")
    else:
        result_label.configure(text=f"The result is Critically Low")
        print("Critically Low")    

root = tk.Tk()
root.title("Dengue Detection")

entries = []
num_inputs = 15



label = tk.Label(root, text=f"Age")
label.grid(row=0, column=1)
label = tk.Label(root, text=f"Hemoglobin")
label.grid(row=1, column=1)
label = tk.Label(root, text=f"RBC")
label.grid(row=2, column=1)
label = tk.Label(root, text=f"PCV")
label.grid(row=3, column=1)
label = tk.Label(root, text=f"MCV")
label.grid(row=4, column=1)
label = tk.Label(root, text=f"MCH")
label.grid(row=5, column=1)
label = tk.Label(root, text=f"MCHC")
label.grid(row=6, column=1)
label = tk.Label(root, text=f"RCBW")
label.grid(row=7, column=1)
label = tk.Label(root, text=f"Platelet")
label.grid(row=8, column=1)
label = tk.Label(root, text=f"Leucocyte")
label.grid(row=9, column=1)
label = tk.Label(root, text=f"Neutrophils")
label.grid(row=10, column=1)
label = tk.Label(root, text=f"Lymphocytes")
label.grid(row=11, column=1)
label = tk.Label(root, text=f"Monocytes")
label.grid(row=12, column=1)
label = tk.Label(root, text=f"Eosinophils")
label.grid(row=13, column=1)
label = tk.Label(root, text=f"Basophils")
label.grid(row=14, column=1)

for i in range(num_inputs):
    entry = tk.Entry(root)
    entry.grid(row=i, column=2)
    entries.append(entry)

button = tk.Button(root, text="Add Inputs", command=add_inputs)
button.grid(row=num_inputs, column=1, columnspan=2)
result_label = tk.Label(root, text="")
result_label.grid(row=num_inputs+1, column=1, columnspan=2)
_img1 = Image.open("1.jpg")
_img1 = _img1.resize((50, 50))
_img1=ImageTk.PhotoImage(_img1)
label1=tk.Label(image=_img1)
label1.image = _img1
label1.place(x=70, y=370)
root.minsize(width=250, height=430)
root.geometry("250x430")
root.configure(background="#dbd8d7")
root.mainloop()
