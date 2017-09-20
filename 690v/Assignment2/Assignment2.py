__author__ = "Siddharth Chandrasekaran"
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "schandraseka@umass.edu"

from bokeh.core.properties import field
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (ColumnDataSource, HoverTool, SingleIntervalTicker, Slider, CategoricalColorMapper, Button,)
from bokeh.palettes import Plasma256
from bokeh.plotting import figure	
from bokeh.io import output_notebook
from bokeh.charts import HeatMap, show, bins
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider
from functools import partial
from bokeh.models.widgets import DataTable
from bokeh.models.glyphs import HBar
#Section -1 Visualizing the data
original_data = pd.read_csv("Wholesalecustomersdata.csv")
missing_data = pd.read_csv("Wholesalecustomersdatamissing.csv")
original_data.Channel = original_data.Channel.astype(str)
original_data.Region = original_data.Region.astype(str)
selected = 'Milk'
selected_data = original_data
selected_data['Data'] = selected_data[selected]
group = selected_data.groupby(('Channel','Region'))
source = ColumnDataSource(group)

index_cmap = factor_cmap('Channel_Region', palette=Spectral6, factors=sorted(original_data.Region.unique()), end=1)
p1 = figure(plot_width=700, plot_height=300, title="Milk mean order value V/S Channel-Region",
           x_range=group)
p1.vbar(x='Channel_Region', top='Data_mean', width=0.5, source=source,
       line_color="white", fill_color=index_cmap, )

p1.add_tools(HoverTool(tooltips=[("Mean Order Quantity", "@Data_mean"), ("Channel, Region", "@Channel_Region")]))

p1.xaxis.axis_label = "Channel,Region"
p1.yaxis.axis_label = "Mean of the order value of a Product"

def sourcemodify(attr, old, new, foo):
	val = select.value
	p1.title.text = select.value + " mean order value V/S Channel-Region"
	foo['Data'] = foo[val]
	group = foo.groupby(('Channel','Region'))
	source.data = ColumnDataSource(group).data

select = Select(title="Product:", value=selected, options=["Milk", "Grocery", "Detergents_Paper", "Delicassen"])
select.on_change('value',partial(sourcemodify, foo=selected_data)) 

#Looks like one entry from Milk, two from Grocery, two from DetergentsPaper and one from Delicassen are missing
missing_rows = missing_data.loc[missing_data.isnull().any(axis=1), :]

#Section 2
#We'll be visualizing the following techniques to fill in missing data
#1. Replacing with global constant
#2. Mean imputation
#3. Hot deck imputation - by finding the nearest neighbor (k=1)
#4. Imputation by fitting a linear model

def handle_categorical_data(data):
    return pd.DataFrame(binarize(data.as_matrix(columns=None), [0,1]), columns=['Channel', 'Region','Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen', 'one', 'two', 'three', 'four', 'five'])

def binarize(data, cols, miss_col = False):
    for col in cols:
        uniq_val, index = np.unique(data[:, col], return_inverse=True)
        data = np.column_stack((data, np.eye(uniq_val.shape[0],
dtype=int)[index]))
    val_cols = [n for n in range(0,data.shape[1]) if n not in cols]
    return data

#Dropped the rows with missing data
missing_data_imputed0 = missing_data.dropna()


processed_data = handle_categorical_data(missing_data)

#Filling the missing values with -1
missing_data_imputed1 = missing_data.loc[processed_data.isnull().any(axis=1),:].fillna(-1)
index = missing_data_imputed1.index

#Filling the missing value with mean of that column
missing_data_imputed2 = missing_data.apply(lambda x: x.fillna(x.mean()),axis=0)
missing_data_imputed2 = missing_data_imputed2.loc[index]


missing_rowsknn = processed_data.loc[processed_data.isnull().any(axis=1),:]

#Filling the missing value using hot deck imputation. Replace the value with the nearest neighbor's value
for missingcol in ['Milk', 'Grocery', 'Detergents_Paper', 'Delicassen']:
    missing_data_imputed4 = processed_data
    missing_data_imputed4 = missing_data_imputed4.dropna()
    features = list(missing_data_imputed4.columns.values)
    features.remove(missingcol)
    features.remove('Channel')
    features.remove('Region')
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(missing_data_imputed4[features],missing_data_imputed4[missingcol])
    predictions = knn.predict(processed_data.loc[processed_data[missingcol].isnull(),:][features])
    predictions = predictions[0]
    missing_rowsknn = missing_rowsknn.set_value(processed_data.loc[processed_data[missingcol].isnull(),:].index,missingcol,predictions)
missing_data_imputed3 = missing_rowsknn

#Filling the missing value using a mutivariate linear model
missing_rowslr = processed_data.loc[processed_data.isnull().any(axis=1),:]
for missingcol in ['Milk', 'Grocery', 'Detergents_Paper', 'Delicassen']:
    missing_data_imputed3 = processed_data
    missing_data_imputed3 = missing_data_imputed3.dropna()
    features = list(missing_data_imputed3.columns.values)
    features.remove(missingcol)
    features.remove('Channel')
    features.remove('Region')
    lreg = LinearRegression()
    lreg.fit(missing_data_imputed3[features],missing_data_imputed3[missingcol])
    #display(processed_data.loc[processed_data[missingcol].isnull(),:][features])
    predictions = lreg.predict(processed_data.loc[processed_data[missingcol].isnull(),:][features])
    predictions = predictions[0]
    missing_rowslr = missing_rowslr.set_value(processed_data.loc[processed_data[missingcol].isnull(),:].index,missingcol,predictions)
missing_data_imputed4 = missing_rowslr

original_values = original_data.loc[index]
#Now dataset is missing_data_imputed4, missing_data_imputed3, missing_data_imputed2, missing_data_imputed1 and the actual values are at
#original_data 
orglist = []
meth1list = []
meth2list = []
meth3list = []
meth4list = []
for i in ['Milk', 'Grocery', 'Detergents_Paper', 'Delicassen']:
	for j in missing_data[missing_data[i].isnull()].index:
		orglist.append(original_data.iloc[j][i])
		meth1list.append(missing_data_imputed1.loc[j,i])
		meth2list.append(missing_data_imputed2.loc[j,i])
		meth3list.append(missing_data_imputed3.iloc[j][i])
		meth4list.append(missing_data_imputed4.loc[j,i])
cols = ['Milk', 'Grocery1', 'Grocery2','Detergents_Paper1','Detergents_Paper2','Delicassen']
data = {}
data['cols'] = cols
data['org'] = orglist
data['Method 1 (GC)'] = meth1list
data['Method 2 (Mean)'] = meth2list
data['Method 3 (KNN)'] = meth3list
data['Method 4 (LR)'] = meth4list

techs = ['org', 'meth1', 'meth2', 'meth3', 'meth4']

#Initializing it for method 1. This gets controlled by the drop box selector
source1 = ColumnDataSource(dict(y=cols, right=orglist, compare = meth1list))


p2 = figure(plot_width=700, plot_height=300, title="Estimated Values of Method 1 (GC) Vs Predicted Value Bar Chart",
           y_range=cols, x_range = (-2,10000))
glyph = HBar(y="y", right="right", left=0, height=0.5, fill_color="#99d594", fill_alpha = 0.6)
p2.add_glyph(source1, glyph)

glyph1 = HBar(y="y", right="compare", left=0, height=0.5, fill_color="#d53e4f", fill_alpha = 0.6)
p2.add_glyph(source1, glyph1)
p2.xaxis.axis_label = "Money Spent"
p2.yaxis.axis_label = "Missing value's Product"


def source1modify(attr, old, new, foo):
	val = select1.value
	p2.title.text = "Estimated Values of " +select1.value + " Vs Predicted Value Bar Chart"
	source1.data = ColumnDataSource(dict(y=cols, right=orglist, compare = foo[select1.value])).data

select1 = Select(title="Product:", value=selected, options=["Method 1 (GC)", "Method 2 (Mean)", "Method 3 (KNN)", "Method 4 (LR)"])
select1.on_change('value',partial(source1modify, foo=data)) 

layout = layout([
    [p1,select],
    [p2,select1]
])
curdoc().add_root(layout)
curdoc().title = "690V Assignment - Whole Customer Dataset"
show(layout)
