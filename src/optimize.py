#!/usr/bin/env python3
import pandas

def bit_array_to_int(row):
    for i, bit in enumerate(row):
        if bit:
            return i + 1
    return 0

def optimize():
    df = pandas.read_csv('./covtype.data', header=None)
    df['clazz']                        = df.iloc[:,-1]
    print('optimizing Wilderness_Area')
    df['Wilderness_Area']                  = df.iloc[:, 10:14].apply(bit_array_to_int, axis=1) 
    print('optimizing Soil_Type')
    df['Soil_Type']                        = df.iloc[:, 14:54].apply(bit_array_to_int, axis=1) 
    df['Elevation']                        = df.iloc[:,0]
    df['Aspect']                           = df.iloc[:,1]
    df['Slope']                            = df.iloc[:,2]
    df['Horizontal_Distance_To_Hydrology'] = df.iloc[:,3]
    df['Vertical_Distance_To_Hydrology']   = df.iloc[:,4]
    df['Horizontal_Distance_To_Roadways']  = df.iloc[:,5]
    df['Hillshade_9am']  = df.iloc[:,6]
    df['Hillshade_Noon']  = df.iloc[:,7]
    df['Hillshade_3pm']  = df.iloc[:,8]
    print('Combining Hillshade using their mean')
    df['Hillshade_Mean'] = df[[ 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].mean(axis=1)
    df['Horizontal_Distance_To_Fire_Points'] = df.iloc[:,9]
    df_opt_log = df[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area', 'Soil_Type', 'clazz']]
    df_opt = df[['Elevation', 'Slope', 'Hillshade_Mean','clazz']]
    df_opt_log.to_csv('./covtype.data.opt.log.csv', index=False)
    df_opt.to_csv('./covtype.data.opt.csv', index=False)

if __name__ == "__main__":
    optimize()