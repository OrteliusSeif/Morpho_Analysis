# %%
from osgeo import gdal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os 
import rasterio 
from rasterio.plot import show
import geopandas as gpd


# %%


# %%
import pandas as pd
from windrose import WindroseAxes



# %%
#Prints the current working directory
os.getcwd()

# %%
# To set the working directory
os.chdir("D:\Tokaj_Morpho_Tectonic\Python")

# %%
os.getcwd()

# %%
DEM = r"D:\Tokaj_Morpho_Tectonic\Polsar_images\DEM_Creation_Polsar\DEM_Tokaj_Polsar.tif"
gdal_data = gdal.Open(DEM)
gdal_band = gdal_data.GetRasterBand(1)
nodataval = gdal_band.GetNoDataValue()

# Convert to numpy array
data_array = gdal_data.ReadAsArray().astype(np.float)
data_array

# replace missing values if necessary 
if np.any(data_array == nodataval): data_array[data_array == nodataval] = np.nan

# %% [markdown]
# Visualize Data with MAtplotlib

# %%
#Plot out data with Matplotlib's 'ColorPlot'
fig = plt.figure(figsize = (15, 9))
ax = fig.add_subplot(111)
plt.contour(data_array, cmap = "viridis", 
            levels = list(range(0, 2000, 20)))
plt.title("Tokaj DEM ")
cbar = plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# %% [markdown]
#  Open the Raster with Rasterio

# %%
Import Lineaments (North East and South East directional filtered data)


# %%
#Import vector data
NE_Directional = gpd.read_file(r"D:\Tokaj_Morpho_Tectonic\Derived_DEM_TIFF\Vector_From_Raster\Simplified_SHP\North_East_Directional_NP1_Douglas_Peucker100T_Exploaded.shp")
SE_Directional = gpd.read_file(r"D:\Tokaj_Morpho_Tectonic\Derived_DEM_TIFF\Vector_From_Raster\Simplified_SHP\South_East_Directional_NP1_Douglas_Peucker100T_Exploded.shp")



# %%
print(NE_Directional.head())
print(NE_Directional.geom_type)


# %%

# View the CRS of the NE_Directional
print(NE_Directional.crs)

# View the spatial extent 
# the data type
type(NE_Directional)
print(NE_Directional.total_bounds)

# How many features are in the shapefile
print(NE_Directional.shape)

# Reproject the Vector data to the same CRS of the Study_Area_DEM


# Reproject the aoi to the same CRS as the state_boundary_use object
#Reproj_NE_Directional= NE_Directional.to_crs(23700)

# View CRS of new reprojected layer
#print(Reproj_NE_Directional.total_bounds)
#print('Reproj_NE_Directional crs: ',Reproj_NE_Directional.crs)




#This is the plot function
fig, ax = plt.subplots(figsize=(10,10))

#plot the data using geopandas.plot() method
NE_Directional.plot(ax=ax, color ='red', linewidth=1)
SE_Directional.plot(ax=ax, color='blue', linewidth=1)
#ax.plot(range(485000,570000,5000), range(532000,540000,1000 ))
ax.ticklabel_format(style='plain')
plt.legend(["Filtered lineaments with NE directional filter", "Filtered lineaments with SE directional filter"], loc="lower right")
plt.show()


# %%
# Open a raster file
Tokaj = r'D:\Tokaj_Morpho_Tectonic\Study_area_DEM.tif'
img = rasterio.open(Tokaj)
print(img.crs)
print(img.count)
fig, ax = plt.subplots(figsize=(10,10))
ax.text(805000, 350000, 'zone I', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 10})
ax.text(800000, 370000, 'zone II', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 10})
ax.text(820000, 350000, 'zone III', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 10})

#NE_Directional.plot(ax=ax, color="black")
show(img, ax=ax)


# %% [markdown]
# # Windrose plot for lineaments direction

# %%
NE direction / First Zone

# %%
#Import Tabular Data from a CSV file into a Pandas
import pandas as pd
from windrose import WindroseAxes
import matplotlib.cm as cm


Azimuth_NE_Region1 = pd.read_csv(r'D:\Tokaj_Morpho_Tectonic\Excel_Files\North_East_Directional_NP1_RegionOne_CSV.csv')

print(type(Azimuth_NE_Region1))
#print(Azimuth_NE_Region1.head())
#display(Azimuth_NE_Region1)

dir = Azimuth_NE_Region1['Azi_deg']
len = Azimuth_NE_Region1 ['Length']

#Azi_rad_column = Azimuth_NE_Region1.loc[:, "Azi_rad"]
#Azi_rad = Azi_rad_column.values
#print(Azi_rad)


#Length_Column = Azimuth_NE_Region1.loc[:, "Length"]
#Length = Length_Column.values


ax1 = WindroseAxes.from_ax()
ax1.bar(dir,len, opening=0.8, edgecolor="k")
ax1.set_legend(loc = "lower right")
ax1.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
ax1.set_title('Filtered lineaments with NE directional filter: Zone number 1', fontweight ="bold", loc="Center")

#help(ax.bar)

# %%
NE Direction / Third zone 

# %%
Azimuth_NE_Region3 = pd.read_csv (r'D:\Tokaj_Morpho_Tectonic\Excel_Files\North_East_Directional_NP1_RegionThree_CSV.csv')
#display(Azimuth_NE_Region3)

dir3 = Azimuth_NE_Region3['Azi_deg']
len3 = Azimuth_NE_Region3 ['Length']


ax3 = WindroseAxes.from_ax()
ax3.set_title('Filtered lineaments with NE directional filter: Zone number 3', fontweight ="bold", loc="Center")
ax3.bar(dir3,len3, opening=0.8, edgecolor="k")
ax3.set_legend(loc = "lower right")
ax3.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])


# %%
SE Direction / First zone 

# %%
Azimuth_SE_Region1 = pd.read_csv (r'D:\Tokaj_Morpho_Tectonic\Excel_Files\South_East_Directional_NP1_RegionOne_CSV.csv')
display(Azimuth_SE_Region1)

dir11 = Azimuth_SE_Region1['Azi_deg']
len11 = Azimuth_SE_Region1 ['Length']


ax11 = WindroseAxes.from_ax()
ax11.set_title('Filtered lineaments with SE directional filter: Zone number 1', fontweight ="bold", loc="Center")
ax11.bar(dir11,len11, opening=0.8, edgecolor="k")
ax11.set_legend(loc = "lower right")
ax11.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])


# %%
SE Direction / Third zone 

# %%
Azimuth_SE_Region3 = pd.read_csv (r'D:\Tokaj_Morpho_Tectonic\Excel_Files\South_East_Directional_NP1_RegionThree_CSV.csv')
display(Azimuth_SE_Region3)

dir33 = Azimuth_SE_Region3['Azi_deg']
len33 = Azimuth_SE_Region3 ['Length']


ax33 = WindroseAxes.from_ax()
ax33.set_title('Filtered lineaments with SE directional filter: Zone number 1', fontweight ="bold", loc="Center")
ax33.bar(dir33,len33, opening=0.8, edgecolor="k")
ax33.set_legend(loc = "lower right")
ax33.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])

# %%
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(2,2)
plt.tight_layout(pad=9.0)
ax1 = fig.add_subplot(gs[0,0], projection = "windrose")
ax1.bar(dir,len)
ax1.set_title('NE directional filter: Zone number 1', fontweight ="bold", loc="Center", fontsize=12)


ax11 = fig.add_subplot(gs[0,1], projection = "windrose")
ax11.bar(dir11,len11)
ax11.set_title('NE directional filter: Zone number 3', fontweight ="bold", loc="Center", fontsize=12)


ax3 = fig.add_subplot(gs[1,0], projection = "windrose")
ax3.bar(dir3,len3)
ax3.set_title(' SE directional filter: Zone number 1', fontweight ="bold", loc="Center", fontsize=12)


ax33 = fig.add_subplot(gs[1,1], projection = "windrose")
ax33.bar(dir33,len33)
ax33.set_title(' SE directional filter: Zone number 3', fontweight ="bold", loc="Center", fontsize=12)
plt.subplots_adjust(hspace=0.1, wspace=1.5)

# %% [markdown]
# # Plot Hmax of the processed Faults: Dexter and Sinister for both zones

# %%
Sinister_Region1 = pd.read_csv (r'D:\Tokaj_Morpho_Tectonic\Excel_Files\Faults_West_Tokaj_Sinister_RegionOne.csv')
#display(Azimuth_SE_Region1)

dirS1 = Sinister_Region1['HMAX_deg']
countS1 = Sinister_Region1 ['COUNT']


ax11 = WindroseAxes.from_ax()
ax11.set_title('Hmax direction from the sinistral interpretation: Zone number 1', fontweight ="bold", loc="Center")
ax11.bar(dirS1,countS1, opening=0.8, edgecolor="k")
ax11.set_legend(loc = "lower right")
ax11.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])

# %%
Sinister_Region3 = pd.read_csv (r'D:\Tokaj_Morpho_Tectonic\Excel_Files\Faults_West_Tokaj_Sinister_RegionThree.csv')


dirS3 = Sinister_Region3['HMAX_deg']
countS3 = Sinister_Region3 ['COUNT']


ax11 = WindroseAxes.from_ax()
ax11.set_title('Hmax direction from the sinistral interpretation: Zone number 3', fontweight ="bold", loc="Center")
ax11.bar(dirS3,countS3, opening=0.8, edgecolor="k")
ax11.set_legend(loc = "lower right")
ax11.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])

# %%
Dextral_Region1 = pd.read_csv (r'D:\Tokaj_Morpho_Tectonic\Excel_Files\Faults_West_Tokaj_Dexter_RegionOne.csv')


dirD1 = Dextral_Region1['HMAX_deg']
countD1 = Dextral_Region1 ['COUNT']


ax11 = WindroseAxes.from_ax()
ax11.set_title('Hmax direction from the Dextral interpretation: Zone number 1', fontweight ="bold", loc="Center")
ax11.bar(dirD1,countD1, opening=0.8, edgecolor="k")
ax11.set_legend(loc = "lower right")
ax11.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])

# %%
Dextral_Region3 = pd.read_csv (r'D:\Tokaj_Morpho_Tectonic\Excel_Files\Faults_West_Tokaj_Dexter_RegionThree.csv')


dirD3 = Dextral_Region3['HMAX_deg']
countD3 = Dextral_Region3 ['COUNT']


ax11 = WindroseAxes.from_ax()
ax11.set_title('Hmax direction from the Dextral interpretation: Zone number 3', fontweight ="bold", loc="Center")
ax11.bar(dirD3,countD3, opening=0.8, edgecolor="k")
ax11.set_legend(loc = "lower right")
ax11.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])

# %%
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(2,2)
plt.tight_layout(pad=9.0)
ax1 = fig.add_subplot(gs[0,0], projection = "windrose")
ax1.bar(dirS1,countS1)
ax1.set_title('Hmax direction from the sinistral interpretation: Zone number 1', fontweight ="bold", loc="Center", fontsize=12)


ax11 = fig.add_subplot(gs[0,1], projection = "windrose")
ax11.bar(dirS3,countS3)
ax11.set_title('Hmax direction from the sinistral interpretation: Zone number 3', fontweight ="bold", loc="Center", fontsize=12)


ax3 = fig.add_subplot(gs[1,0], projection = "windrose")
ax3.bar(dirD1,countD1)
ax3.set_title(' Hmax direction from the dextral interpretation: Zone number 1', fontweight ="bold", loc="Center", fontsize=12)


ax33 = fig.add_subplot(gs[1,1], projection = "windrose")
ax33.bar(dirD3,countD3)
ax33.set_title(' Hmax direction from the dextral interpretation: Zone number 3', fontweight ="bold", loc="Center", fontsize=12)
plt.subplots_adjust(hspace=0.1, wspace=3)

# %%
All = [Azimuth_NE_Region1, Azimuth_NE_Region3, Azimuth_SE_Region1 ,Azimuth_SE_Region3, Sinister_Region1, Sinister_Region3
, Dextral_Region1
, Dextral_Region3]
Alltogether = pd.concat(All)
print(Alltogether.head())

# %%
del Alltogether['Azi_rad']

# %%
print(Alltogether.head())

# %%
Alltogether.drop(columns=['ID', 'region','Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'WEIGHT', 'HMAX_ANGLE','HMAX_LENGT', 'S1V', 'Regions'])

# %% [markdown]
# # Data correlation Zone 1

# %%
All_zone1 = [Azimuth_NE_Region1, Azimuth_SE_Region1, Sinister_Region1, Dextral_Region1]
Alltogether_1 = pd.concat(All_zone1)
print(Alltogether_1.head())

# %%
display(Azimuth_NE_Region1)

# %%
display(Azimuth_SE_Region1)

# %%
Region1 = pd.concat([Azimuth_NE_Region1,Azimuth_SE_Region1,Sinister_Region1, Dextral_Region1], axis=1)

#Region1 = Region1.drop(Region1[1, axis = 1)
print(Region1.columns.values)
display(Region1)
#DataFrame. info() gives the colum number, the column name and the type
#Region1.info()

# %%
#Region1.drop(Region1.columns[], axis=1, inplace=True)
print(Region1.head())

# %%
print(Region1.columns.values)
display(Region1)
#Region1.drop(Region1.columns[[1,3,4]], axis=1, inplace=True)

# %%
Here all unnecessary column were removed, generated the row Excel file 

# %%
Region1.drop(Region1.columns[[0,2,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20,21,23,25,28,29,31,32,33,34,36]], axis=1, inplace=True)
display(Region1)

# %%
display(Region1)

# %%
#print out columns of Region1 DataFrame
print(Region1.columns)

# %%
Region1.rename(columns = {'Length':"Length_Azimuth_NE"}, inplace = True)

# %%
print(Region1.columns)

# %%
#Renaming identical columns, since clolumn names were the identical
Region1.columns.values[2] = "Length_Azimuth_SE"
Region1.columns.values[1] = "Azi_NE_deg"
Region1.columns.values[3]  = "Azi_SE_deg"
Region1.columns.values[4]  = "Count_Sinister"
Region1.columns.values[5]  = "HMAX_Sinister_deg"
Region1.columns.values[6]  = "Count_Dexter"
Region1.columns.values[7]  = "HMAX_Dextral_deg"



# %%
print(Region1.columns)

# %%


# %%
#Count NULL values 
#Region1.isnull().sum()

# %%
Coor = Region1.corr()
display(Coor)

# %%
import seaborn as sns
plt.figure(figsize=(10,5))
sns.heatmap(Coor, cmap='YlGnBu')

# %% [markdown]
# # Data Correlation Zone 3

# %%
All_zone3 = [Azimuth_NE_Region3, Azimuth_SE_Region3, Sinister_Region3, Dextral_Region3]
Alltogether_3 = pd.concat(All_zone3)
print(Alltogether_3.head())

# %%
Region3 = pd.concat([Azimuth_NE_Region3, Azimuth_SE_Region3, Sinister_Region3, Dextral_Region3], axis=1)

print(Region3.columns.values)
display(Region3)
Region3.info()

# %%
Region3.drop(Region3.columns[[0,2,4,5,6,7,8,9,10,11,13,15,16,17,18,19,
                              20,21,22,24,25,26,27,29,30,32,33,34,35,37]], axis=1, inplace=True)
display(Region3)

# %%
print(Region3.columns)

# %%
Region3.info()

# %%
#Renaming identical columns
Region3.columns.values[0] = "Length_Azimuth_NE"
Region3.columns.values[1] = "Azi_NE_deg"
Region3.columns.values[2]  = "Length_Azimuth_SE"
Region3.columns.values[3]  = "Azi_SE_deg"
Region3.columns.values[4]  = "Count_Sinister"
Region3.columns.values[5]  = "HMAX_Sinister_deg"
Region3.columns.values[6]  = "Count_Dextral"
Region3.columns.values[7]  = "HMAX_Dextral_deg"
print(Region3.columns)


# %%
Coor3 = Region3.corr()
display(Coor3)

# %%
import seaborn as sns
plt.figure(figsize=(10,5))
sns.heatmap(Coor3, cmap='YlGnBu')

# %%


# %%


# %%