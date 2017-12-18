from bokeh.plotting import *
from bokeh.io import output_file, show
from bokeh.core.properties import value
from bokeh.events import ButtonClick
from bokeh.layouts import widgetbox, layout, row, column
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap, dodge
from bokeh.palettes import Spectral6
from bokeh.models.widgets import *
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from numpy import pi
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import statistics
import graphviz 
from sklearn.feature_selection import SelectPercentile, f_classif

def PrepareLocation():
    locations= dataFrame[sheetColumns[3]];
    data={};
    total=0;
    for location in locations:
        if location in data:
            data[location]= data[location]+1;
        else:
            data[location]=1;
        total=total+1;
    return data,total;

def PrepareGender():
    genders= dataFrame[sheetColumns[5]];
    data={};
    total=0;
    for gender in genders:
        if gender in data:
            data[gender]=data[gender]+1;
        else:
            data[gender]=1;
        total=total+1;
    return data,total;

def PrepareHistology():
    histologies= dataFrame[sheetColumns[6]];
    data={};
    total=0;
    for histology in histologies:
        histologyLower= histology.lower();
        if "squamous" in histologyLower:
            if 'Squamous Cell Carcinoma' in data:
                data['Squamous Cell Carcinoma']=data['Squamous Cell Carcinoma']+1;
            else:
                data['Squamous Cell Carcinoma']=1;
        elif "adenocarcinoma" in histologyLower or "papillary" in histologyLower or "micropapillary" in histologyLower or "mucinous" in histologyLower or "acinar" in histologyLower or "solid"in histologyLower:
            if 'Adenocarcinoma' in data:
                data['Adenocarcinoma']=data['Adenocarcinoma']+1;
            else:
                data['Adenocarcinoma']=1;
        else:
            if 'Unspecified' in data:
                data['Unspecified']=data['Unspecified']+1;
            else:
                data['Unspecified']=1;
        total=total+1;
    return data,total;

def PrepareStage():
    primaries= dataFrame[sheetColumns[8]];
    nodes= dataFrame[sheetColumns[9]];
    mets= dataFrame[sheetColumns[10]];
    primariesData=[0,0,0,0];
    nodesData=[0,0,0,0];
    metsData=[0,0,0,0];
    for primary in primaries:
        primaryLower= primary.lower();
        if "pt0" in primaryLower:
            primariesData[0]=primariesData[0]+1;
        elif "pt1" in primaryLower:
            primariesData[1]=primariesData[1]+1;
        elif "pt2" in primaryLower:
            primariesData[2]=primariesData[2]+1;
        elif "pt3" in primaryLower:
            primariesData[3]=primariesData[3]+1;
            
    for node in nodes:
        nodeLower= node.lower();
        if "pn0" in nodeLower:
            nodesData[0]=nodesData[0]+1;
        elif "pn1" in nodeLower:
            nodesData[1]=nodesData[1]+1;
        elif "pn2" in nodeLower:
            nodesData[2]=nodesData[2]+1;
        elif "pn3" in nodeLower:
            nodesData[3]=nodesData[3]+1;
            
    for met in mets:
        metLower= met.lower();
        if "pm0" in metLower:
            metsData[0]=metsData[0]+1;
        elif "pm1" in metLower:
            metsData[1]=metsData[1]+1;
        elif "pm2" in metLower:
            metsData[2]=metsData[2]+1;
        elif "pm3" in metLower:
            metsData[3]=metsData[3]+1;
    return primariesData,nodesData,metsData;


def PrepareTarget():
    histologies= dataFrame[sheetColumns[6]];
    target=[];
    unspecified=[];
    index=0;
    for histology in histologies:
        histologyLower= histology.lower();
        if "squamous" in histologyLower:
            target.append([1]);
        elif "adenocarcinoma" in histologyLower or "papillary" in histologyLower or "micropapillary" in histologyLower or "mucinous" in histologyLower or "acinar" in histologyLower or "solid"in histologyLower:
            target.append([0]);
        else:
            unspecified.append(index);
        index=index+1;
    target= np.array(target);
    return target,unspecified;

dataFrame = pd.read_excel('Data/Lung3.metadata.xls', sheet_name='Lung3.metadata')
sheetColumns= dataFrame.columns;
dataFrameNameIndexed= dataFrame.set_index(sheetColumns[0]);
sampleSerie= dataFrame[sheetColumns[0]];
sampleList= sampleSerie.tolist();

patientSelect = Select(title="Patient:", value=sampleList[0], options=sampleList);
patientInfo= Div(text="""<br><h1>"""+patientSelect.value+"""</h1><br>
    <div><h2 style='display:inline'>Location: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[3]]+"""</p></div><br>
    <div><h2 style='display:inline'>Organism: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[4]]+"""</p></div><br>
    <div><h2 style='display:inline'>Gender: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[5]]+"""</p></div><br>
    <div><h2 style='display:inline'>Histology: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[6]]+"""</p></div><br>
    <div><h2 style='display:inline'>Tumor Size: </h2><p style='display:inline; font-size:18px'>"""+str(dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[7]])+"""</p></div><br>
    <div><h2 style='display:inline'>Primary Tumor Stage: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[8]]+"""</p></div><br>
    <div><h2 style='display:inline'>Node Stage: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[9]]+"""</p></div><br>
    <div><h2 style='display:inline'>Mets Stage: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[10]]+"""</p></div><br>
    <div><h2 style='display:inline'>Primary/Mets: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[11]]+"""</p></div><br>
    <div><h2 style='display:inline'>Grade: </h2><p style='display:inline; font-size:18px'>"""+str(dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[12]])+"""</p></div><br>
    <div><h2 style='display:inline'>Test Molecule: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[13]]+"""</p></div><br>
    <div><h2 style='display:inline'>Label: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[14]]+"""</p></div><br>
    <div><h2 style='display:inline'>Platform: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[15]]+"""</p></div>""",width=900);
patientLayout = layout([[patientSelect,patientInfo]]);
patientTab= Panel(child=patientLayout, title="Patients");


locationRawData,locationTotal=PrepareLocation();
locationMode=max(locationRawData, key=locationRawData.get);
locationKeys=list(locationRawData.keys());
locationValues=list(locationRawData.values());
locationData = dict(
        data1=locationKeys+['Total','Mode'],
        data2=locationValues+[locationTotal,locationMode],
    )
locationSource = ColumnDataSource(locationData)

locationColumns = [
        TableColumn(field="data1", title='Location'),
        TableColumn(field="data2", title='Value'),
    ]
locationTable = DataTable(source=locationSource, columns=locationColumns)

locationHistogramSource = ColumnDataSource(data=dict(data1=locationKeys, data2=locationValues));

locationHistogram = figure(x_range=locationKeys, plot_height=350, toolbar_location=None, title="Location")
locationHistogram.vbar(x='data1', top='data2', width=0.9, source=locationHistogramSource, legend="data1",
       line_color='white', fill_color=factor_cmap('data1', palette=Spectral6, factors=locationKeys));

ratio = [locationValue/locationTotal for locationValue in list(locationRawData.values())];
percents=[0];
for r in ratio:
    percents=percents+[percents[-1]+r];
starts = [p*2*pi for p in percents[:-1]];
ends = [p*2*pi for p in percents[1:]];
locationPieChartSource= ColumnDataSource(data=dict(data1=locationKeys, data2=[0]*len(starts), data3=[0]*len(starts), starts=starts, ends=ends));
locationPieChart = figure(x_range=(-1,1), y_range=(-1,1));
locationPieChart.wedge(x='data2', y='data3', radius=1, start_angle='starts', end_angle='ends',source=locationPieChartSource,legend="data1",line_color='white', fill_color=factor_cmap('data1', palette=Spectral6, factors=locationKeys));

locationLayout = layout([locationTable,[locationHistogram,locationPieChart]]);
locationTab= Panel(child=locationLayout, title="Location");


genderRawData,genderTotal=PrepareGender();
genderMode=max(genderRawData, key=genderRawData.get);
genderKeys=list(genderRawData.keys());
genderValues=list(genderRawData.values());
genderData = dict(
        data1=genderKeys+['Total','Mode'],
        data2=genderValues+[genderTotal,genderMode],
    )
genderSource = ColumnDataSource(genderData)

genderColumns = [
        TableColumn(field="data1", title='Gender'),
        TableColumn(field="data2", title='Value'),
    ]
genderTable = DataTable(source=genderSource, columns=genderColumns)

genderHistogramSource = ColumnDataSource(data=dict(data1=genderKeys, data2=genderValues));

genderHistogram = figure(x_range=genderKeys, plot_height=350, toolbar_location=None, title="Gender")
genderHistogram.vbar(x='data1', top='data2', width=0.9, source=genderHistogramSource, legend="data1",
       line_color='white', fill_color=factor_cmap('data1', palette=Spectral6, factors=genderKeys));

ratio = [genderValue/genderTotal for genderValue in list(genderRawData.values())];
percents=[0];
for r in ratio:
    percents=percents+[percents[-1]+r];
starts = [p*2*pi for p in percents[:-1]];
ends = [p*2*pi for p in percents[1:]];
genderPieChartSource= ColumnDataSource(data=dict(data1=genderKeys, data2=[0]*len(starts), data3=[0]*len(starts), starts=starts, ends=ends));
genderPieChart = figure(x_range=(-1,1), y_range=(-1,1));
genderPieChart.wedge(x='data2', y='data3', radius=1, start_angle='starts', end_angle='ends',source=genderPieChartSource,legend="data1",line_color='white', fill_color=factor_cmap('data1', palette=Spectral6, factors=genderKeys));

genderLayout = layout([genderTable,[genderHistogram,genderPieChart]]);
genderTab= Panel(child=genderLayout, title="Gender");


histologyRawData,histologyTotal=PrepareHistology();
histologyMode=max(histologyRawData, key=histologyRawData.get);
histologyKeys=list(histologyRawData.keys());
histologyValues=list(histologyRawData.values());
histologyData = dict(
        data1=histologyKeys+['Total','Mode'],
        data2=histologyValues+[histologyTotal,histologyMode],
    )
histologySource = ColumnDataSource(histologyData)

histologyColumns = [
        TableColumn(field="data1", title='Histology'),
        TableColumn(field="data2", title='Value'),
    ]
histologyTable = DataTable(source=histologySource, columns=histologyColumns)

histologyHistogramSource = ColumnDataSource(data=dict(data1=histologyKeys, data2=histologyValues));

histologyHistogram = figure(x_range=histologyKeys, plot_height=350, toolbar_location=None, title="Histology")
histologyHistogram.vbar(x='data1', top='data2', width=0.9, source=histologyHistogramSource, legend="data1",
       line_color='white', fill_color=factor_cmap('data1', palette=Spectral6, factors=histologyKeys));

ratio = [histologyValue/histologyTotal for histologyValue in list(histologyRawData.values())];
percents=[0];
for r in ratio:
    percents=percents+[percents[-1]+r];
starts = [p*2*pi for p in percents[:-1]];
ends = [p*2*pi for p in percents[1:]];
histologyPieChartSource= ColumnDataSource(data=dict(data1=histologyKeys, data2=[0]*len(starts), data3=[0]*len(starts), starts=starts, ends=ends));
histologyPieChart = figure(x_range=(-1,1), y_range=(-1,1));
histologyPieChart.wedge(x='data2', y='data3', radius=1, start_angle='starts', end_angle='ends',source=histologyPieChartSource,legend="data1",line_color='white', fill_color=factor_cmap('data1', palette=Spectral6, factors=histologyKeys));

histologyLayout = layout([histologyTable,[histologyHistogram,histologyPieChart]]);
histologyTab= Panel(child=histologyLayout, title="Histology");


tumorSizeRawData=dataFrame[sheetColumns[7]].values.tolist();
del tumorSizeRawData[3];
tumorSizeData = {
    'data1' : ["Mean","Median","Standard Deviation","Variance","Min","Max"],
        'data2'   : [statistics.mean(tumorSizeRawData),statistics.median(tumorSizeRawData),statistics.stdev(tumorSizeRawData),statistics.variance(tumorSizeRawData),min(tumorSizeRawData),max(tumorSizeRawData)],
        }

tumorSizeSource = ColumnDataSource(tumorSizeData)

tumorSizeColumns = [
        TableColumn(field="data1", title='Stats'),
        TableColumn(field="data2", title='Value'),
    ]
tumorSizeTable = DataTable(source=tumorSizeSource, columns=tumorSizeColumns,height=250)
tumorSizePlot=figure(plot_width=700, plot_height=400,title="Tumor Size");
tumorSizePlot.line(range(len(tumorSizeRawData)),tumorSizeRawData);
tumorSizeLayout = column(tumorSizeTable,tumorSizePlot);


primaryRawData,nodeRawData,metRawData= PrepareStage();
cancerStage = ['0', '1', '2', '3'];
cancerCharac = ["Primary", "Node", "Mets"];
charaColors = ["#c9d9d3", "#718dbf", "#e84d60"];

stageData = {'Stage' : cancerStage,
        'Primary'   : primaryRawData,
        'Node'   : nodeRawData,
        'Mets'   : metRawData}

stageSource = ColumnDataSource(stageData)

stageColumns = [
        TableColumn(field="Stage", title='Stage'),
        TableColumn(field="Primary", title='Primary'),
        TableColumn(field="Node", title='Node'),
        TableColumn(field="Mets", title='Mets'),
    ]
stageTable = DataTable(source=stageSource, columns=stageColumns,height=250)

stageHistogram = figure(x_range=cancerStage, y_range=(0, 90), plot_height=400, title="Lung Cancer Stage",
           toolbar_location=None, tools="")

stageHistogram.vbar(x=dodge('Stage', -0.25, range=stageHistogram.x_range), top='Primary', width=0.2, source=stageSource,
       color="#c9d9d3", legend=value("Primary"))

stageHistogram.vbar(x=dodge('Stage',  0.0,  range=stageHistogram.x_range), top='Node', width=0.2, source=stageSource,
       color="#718dbf", legend=value("Node"))

stageHistogram.vbar(x=dodge('Stage',  0.25, range=stageHistogram.x_range), top='Mets', width=0.2, source=stageSource,
       color="#e84d60", legend=value("Mets"))

stageHistogram.x_range.range_padding = 0.1
stageHistogram.xgrid.grid_line_color = None
stageHistogram.legend.location = "top_right"
stageHistogram.legend.orientation = "horizontal"

stageLayout = layout([stageTable,stageHistogram]);


targetArray,unspecifiedPatients= PrepareTarget();
inputFrame = pd.read_csv('Data/GSE58661_RAW/Expressions.csv',sep=';');
featureArray= inputFrame.values;
featureArray= featureArray.transpose();
featureIndices = np.arange(featureArray.shape[-1]);
featureNames= featureArray[0,:];
featureNamesList=featureNames.tolist();
featureArray= featureArray[1:,:];
featureArray= np.delete(featureArray, unspecifiedPatients, axis=0);
targetArray= targetArray.ravel();
selector = SelectPercentile(f_classif, percentile=250*100/60607);
selector.fit(featureArray, targetArray);
scores = -np.log10(selector.pvalues_);
scores /= scores.max();
featureIndices=featureIndices.reshape(1,-1);
selectedFeatureIndices=selector.transform(featureIndices);
selectedFeatureNames=featureNames[selectedFeatureIndices];
scores=scores.reshape(1,-1);
selectedScores= selector.transform(scores);
selectedFeatureNamesList=selectedFeatureNames.tolist()[0];
featuresNumberSlider = Slider(start=0, end=1000, value=250, step=50, title="Number of Features")
probeSelect = Select(title="Probes:", value=selectedFeatureNamesList[0], options=selectedFeatureNamesList);
probeIndex= featureNamesList.index(selectedFeatureNamesList[0]);
probeData = dict(
        data1=featureArray[:,probeIndex].tolist(),
    )
probeSource = ColumnDataSource(probeData)
probeColumns = [
        TableColumn(field="data1", title='Value'),
    ]
probeTable = DataTable(source=probeSource, columns=probeColumns, width=275,height=650);
probeHistogram = figure(plot_width=700, plot_height=400)
probeHistogram.vbar(x=range(80), width=0.8, bottom=0,
       top=featureArray[:,probeIndex].tolist(), color="firebrick");
probeInfo= Div(text="""<br><h1>"""+selectedFeatureNamesList[0]+"""</h1><br>
    <div><h2 style='display:inline'>P Value: </h2><p style='display:inline; font-size:18px'>"""+str(scores[0,probeIndex])+"""</p></div><br>
    <div><h2 style='display:inline'>Mean: </h2><p style='display:inline; font-size:18px'>"""+str(statistics.mean(featureArray[:,probeIndex].tolist()))+"""</p></div><br>
    <div><h2 style='display:inline'>Max: </h2><p style='display:inline; font-size:18px'>"""+str(np.amax(featureArray[:,probeIndex]))+"""</p></div><br>
    <div><h2 style='display:inline'>Third Quartile: </h2><p style='display:inline; font-size:18px'>"""+str(np.percentile(featureArray[:,probeIndex], 75))+"""</p></div><br>
    <div><h2 style='display:inline'>Median: </h2><p style='display:inline; font-size:18px'>"""+str(statistics.median(featureArray[:,probeIndex].tolist()))+"""</p></div><br>
    <div><h2 style='display:inline'>First Quartile: </h2><p style='display:inline; font-size:18px'>"""+str(np.percentile(featureArray[:,probeIndex], 25))+"""</p></div><br>
    <div><h2 style='display:inline'>Min: </h2><p style='display:inline; font-size:18px'>"""+str(np.amin(featureArray[:,probeIndex]))+"""</p></div>""",width=350);
univariateLayout = row(column(probeSelect,featuresNumberSlider,probeInfo),probeHistogram,probeTable);
univariateTab= Panel(child=univariateLayout, title="Univariate Analysis");


attributeSelect=Select(title="Attribute:", value='Location', options=['Location','Gender','Histology','Tumor Size','Stage'],width=275);
attributeLayout = row(attributeSelect,locationLayout);
attributeTab= Panel(child=attributeLayout, title="Attribute");


dataSplitSlider = Slider(start=0, end=100, value=70, step=5, title="Data Split")
trainButton = Button(label="Train", button_type="success")
confusionData = dict(
        data1=['Actual Adenocarcinoma:','Actual Squamous Carcinoma:'],
        data2=[0,0],
        data3=[0,0],
    );
confusionSource = ColumnDataSource(confusionData);
confusionColumns = [
        TableColumn(field="data1", title=''),
        TableColumn(field="data2", title='Predicted Adenocarcinoma:'),
        TableColumn(field="data3", title='Predicted Squamous Carcinoma:'),
    ];
confusionTable = DataTable(source=confusionSource, columns=confusionColumns, width=300,height=200);
decisionTreeInfo= Div(text=""" """,width=700);
space350Wide= Div(text=""" """,width=350);
rocPlot = figure(plot_width=400, plot_height=400);
rocPlot.line([0,1], [0,1], line_width=1.5);
decisionTreeLayout = row(column(dataSplitSlider,trainButton,decisionTreeInfo),column(confusionTable,rocPlot));
decisionTreeTab= Panel(child=decisionTreeLayout, title="Decision Tree");


clinicalData = dict(
        data1=dataFrame[sheetColumns[0]],
        data2=dataFrame[sheetColumns[1]],
        data3=dataFrame[sheetColumns[2]],
        data4=dataFrame[sheetColumns[3]],
        data5=dataFrame[sheetColumns[4]],
        data6=dataFrame[sheetColumns[5]],
        data7=dataFrame[sheetColumns[6]],
        data8=dataFrame[sheetColumns[7]],
        data9=dataFrame[sheetColumns[8]],
        data10=dataFrame[sheetColumns[9]],
        data11=dataFrame[sheetColumns[10]],
        data12=dataFrame[sheetColumns[11]],
        data13=dataFrame[sheetColumns[12]],
        data14=dataFrame[sheetColumns[13]],
        data15=dataFrame[sheetColumns[14]],
        data16=dataFrame[sheetColumns[15]],
    )
clinicalSource = ColumnDataSource(clinicalData)

clinicalColumns = [
        TableColumn(field="data1", title=sheetColumns[0]),
        TableColumn(field="data2", title=sheetColumns[1]),
        TableColumn(field="data3", title=sheetColumns[2]),
        TableColumn(field="data4", title=sheetColumns[3]),
        TableColumn(field="data5", title=sheetColumns[4]),
        TableColumn(field="data6", title=sheetColumns[5]),
        TableColumn(field="data7", title=sheetColumns[6]),
        TableColumn(field="data8", title=sheetColumns[7]),
        TableColumn(field="data9", title=sheetColumns[8]),
        TableColumn(field="data10", title=sheetColumns[9]),
        TableColumn(field="data11", title=sheetColumns[10]),
        TableColumn(field="data12", title=sheetColumns[11]),
        TableColumn(field="data13", title=sheetColumns[12]),
        TableColumn(field="data14", title=sheetColumns[13]),
        TableColumn(field="data15", title=sheetColumns[14]),
        TableColumn(field="data16", title=sheetColumns[15]),
    ]
clinicalTable = DataTable(source=clinicalSource, columns=clinicalColumns, width=1350, height=650)
tableLayout = layout([clinicalTable]);
tableTab= Panel(child=tableLayout, title="Table");

webInterfaceLayout=Tabs(tabs=[patientTab,attributeTab,univariateTab,decisionTreeTab,tableTab]);
curdoc().add_root(webInterfaceLayout);

def patientSelectHandler(attr, old, new):
    patientInfo.text="""<br><h1>"""+patientSelect.value+"""</h1><br>
    <div><h2 style='display:inline'>Location: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[3]]+"""</p></div><br>
    <div><h2 style='display:inline'>Organism: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[4]]+"""</p></div><br>
    <div><h2 style='display:inline'>Gender: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[5]]+"""</p></div><br>
    <div><h2 style='display:inline'>Histology: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[6]]+"""</p></div><br>
    <div><h2 style='display:inline'>Tumor Size: </h2><p style='display:inline; font-size:18px'>"""+str(dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[7]])+"""</p></div><br>
    <div><h2 style='display:inline'>Primary Tumor Stage: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[8]]+"""</p></div><br>
    <div><h2 style='display:inline'>Node Stage: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[9]]+"""</p></div><br>
    <div><h2 style='display:inline'>Mets Stage: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[10]]+"""</p></div><br>
    <div><h2 style='display:inline'>Primary/Mets: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[11]]+"""</p></div><br>
    <div><h2 style='display:inline'>Grade: </h2><p style='display:inline; font-size:18px'>"""+str(dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[12]])+"""</p></div><br>
    <div><h2 style='display:inline'>Test Molecule: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[13]]+"""</p></div><br>
    <div><h2 style='display:inline'>Label: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[14]]+"""</p></div><br>
    <div><h2 style='display:inline'>Platform: </h2><p style='display:inline; font-size:18px'>"""+dataFrameNameIndexed.loc[patientSelect.value,sheetColumns[15]]+"""</p></div>""";

def attributeSelectHandler(attr, old, new):
    'Location','Gender','Histology'
    if(new=='Location'):
        attributeLayout.update(children=[attributeSelect,locationLayout]);
    elif(new=='Gender'):
        attributeLayout.update(children=[attributeSelect,genderLayout]);
    elif(new=='Histology'):
        attributeLayout.update(children=[attributeSelect,histologyLayout]);
    elif(new=='Tumor Size'):
        attributeLayout.update(children=[attributeSelect,tumorSizeLayout]);
    elif(new=='Stage'):
        attributeLayout.update(children=[attributeSelect,stageLayout]);
        
def featuresNumberSliderHandler(attr, old, new):
    selector = SelectPercentile(f_classif, percentile=new*100/60607);
    selector.fit(featureArray, targetArray);
    selectedFeatureIndices=selector.transform(featureIndices);
    selectedFeatureNames=featureNames[selectedFeatureIndices];
    selectedScores= selector.transform(scores);
    selectedFeatureNamesList=selectedFeatureNames.tolist()[0];
    probeSelect.update(title="Probes:", value=selectedFeatureNamesList[0], options=selectedFeatureNamesList);

def probeSelectHandler(attr, old, new):
    probeIndex= featureNamesList.index(new);
    probeHistogram.renderers.clear();
    probeHistogram.vbar(x=range(80), width=0.8, bottom=0,
       top=featureArray[:,probeIndex].tolist(), color="firebrick");
    probeData = dict(
        data1=featureArray[:,probeIndex].tolist(),
    )
    probeSource = ColumnDataSource(probeData);
    probeTable.source=probeSource;
    probeInfo.text="""<br><h1>"""+selectedFeatureNamesList[0]+"""</h1><br>
    <div><h2 style='display:inline'>P Value: </h2><p style='display:inline; font-size:18px'>"""+str(scores[0,probeIndex])+"""</p></div><br>
    <div><h2 style='display:inline'>Mean: </h2><p style='display:inline; font-size:18px'>"""+str(statistics.mean(featureArray[:,probeIndex].tolist()))+"""</p></div><br>
    <div><h2 style='display:inline'>Max: </h2><p style='display:inline; font-size:18px'>"""+str(np.amax(featureArray[:,probeIndex]))+"""</p></div><br>
    <div><h2 style='display:inline'>Third Quartile: </h2><p style='display:inline; font-size:18px'>"""+str(np.percentile(featureArray[:,probeIndex], 75))+"""</p></div><br>
    <div><h2 style='display:inline'>Median: </h2><p style='display:inline; font-size:18px'>"""+str(statistics.median(featureArray[:,probeIndex].tolist()))+"""</p></div><br>
    <div><h2 style='display:inline'>First Quartile: </h2><p style='display:inline; font-size:18px'>"""+str(np.percentile(featureArray[:,probeIndex], 25))+"""</p></div><br>
    <div><h2 style='display:inline'>Min: </h2><p style='display:inline; font-size:18px'>"""+str(np.amin(featureArray[:,probeIndex]))+"""</p></div>""";

def trainButtonHandler():
    if(dataSplitSlider.value==100):
        decisionTree = tree.DecisionTreeClassifier()
        decisionTree = decisionTree.fit(selector.transform(featureArray),targetArray);
        predictionArray= decisionTree.predict(selector.transform(featureArray));
        falsePositiveRate, truePositiveRate, _ = roc_curve(targetArray, predictionArray)
        rocAreaUnderCover = auc(falsePositiveRate, truePositiveRate);
        confusionMatrix = confusion_matrix(targetArray, predictionArray);
    else:
        testRatio = (100-dataSplitSlider.value)/100;
        featureArrayTrain, featureArrayTest, targetArrayTrain, targetArrayTest = train_test_split(selector.transform(featureArray), targetArray, test_size=testRatio,random_state=0);
        decisionTree = tree.DecisionTreeClassifier();
        decisionTree = decisionTree.fit(featureArrayTrain,targetArrayTrain);
        predictionArray= decisionTree.predict(featureArrayTest);
        falsePositiveRate, truePositiveRate, _ = roc_curve(targetArrayTest, predictionArray)
        rocAreaUnderCover = auc(falsePositiveRate, truePositiveRate);
        confusionMatrix = confusion_matrix(targetArrayTest, predictionArray);
    confusionData = dict(
        data1=['Actual Adenocarcinoma:','Actual Squamous Carcinoma:'],
        data2 = confusionMatrix[:,0].tolist(),
        data3 = confusionMatrix[:,1].tolist(),
    );
    confusionSource = ColumnDataSource(confusionData);
    confusionTable.source=confusionSource;
    rocPlot.renderers.clear();
    rocPlot.line(falsePositiveRate.tolist(),truePositiveRate.tolist(), line_width=3, color='orange');
    rocPlot.line([0,1], [0,1], line_width=1.5);

    n_nodes = decisionTree.tree_.node_count
    children_left = decisionTree.tree_.children_left
    children_right = decisionTree.tree_.children_right
    feature = decisionTree.tree_.feature
    threshold = decisionTree.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    decisionTreeString="""<br><p>The binary tree structure has """+str(n_nodes)+""" nodes and has
          the following tree structure:</p>""";
    for i in range(n_nodes):
        if is_leaves[i]:
            decisionTreeString=decisionTreeString+"""<p>node="""+str(i)+""" leaf node.</p>""";
        else:
            decisionTreeString=decisionTreeString+"""<p>node="""+str(i)+"""
            test node: go to node """+str(children_left[i])+""" if """+selectedFeatureNamesList[i]+""" <= """+str(threshold[i])+"""
            else to node"""+str(children_right[i])+""".</p>""";
    print(decisionTreeString);
    decisionTreeInfo.text=decisionTreeString;
    
patientSelect.on_change("value", patientSelectHandler);
attributeSelect.on_change("value", attributeSelectHandler); 
featuresNumberSlider.on_change("value", featuresNumberSliderHandler);
probeSelect.on_change("value", probeSelectHandler);
trainButton.on_click(trainButtonHandler)
