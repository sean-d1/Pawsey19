import pandas as pd
import numpy as np


def frame_filler(df, name: str, size: float):
    data = pd.read_csv(name + ".csv", na_values=[""])

    if name[:5] == "Alice":
        hum_name = name[:-1] + "Hum.txt"
    else:
        hum_name = name + "Hum.txt"

    test = [i for i in range(0,35)]
    test.extend([36,37])
    hum = pd.read_csv(hum_name, skiprows = test, delim_whitespace=True)

    data.rename(columns={"PV Yield (kWh)" : "Yield", "Irradiance (W/m2)" : "Irradiance",
                         "Nearest BOM station temperature (C)" : "Temperature"}, inplace=True)

    data["Size"] = size 

    data['Timestamp_new'] = pd.to_datetime(data['Timestamp'], format = '%Y-%m-%d %H:%M:%S', errors = 'coerce').dt.to_period("H")
    data["Hour"] = data["Timestamp_new"].apply(lambda x : x.hour)
    data["Month"] = data["Timestamp_new"].apply(lambda x : x.month)

    hours = ["H" + str(i) for i in range(0, 24)]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for i, name in enumerate(hours):
        data[name] = data["Hour"].apply(lambda x: 1 if x == i else 0)
    
    for i, name in enumerate(months):
        data[name] = data["Month"].apply(lambda x: 1 if x == i + 1 else 0)

    data["Date"] =  pd.to_datetime(data['Timestamp'], format = '%Y-%m-%d %H:%M:%S', errors = 'coerce').dt.to_period("D")
    hum["Date"] = pd.to_datetime(hum["Date2"], format = "%d/%m/%Y", errors='coerce').dt.to_period("D")

    data.loc[data["Yield"] < 0, "Yield"] = np.nan
    data.loc[data["Yield"] > size] = np.nan
    data.loc[data["Temperature"] > 60, "Temperature"] = np.nan
    data.loc[data["Temperature"] < -30, "Temperature"] = np.nan
    data.loc[data["Irradiance"] < 0] = np.nan 

    data.dropna(subset = ["Yield", "Timestamp", "Irradiance", "Temperature"], inplace = True)
    all_data = pd.merge(data, hum, how="left", on = 'Date')

    all_data["Humidity"] = all_data.apply(lambda x: (x["Temperature"] - x["T.Min"]) * ((x["RHmaxT"] - x["RHminT"]) / (x["T.Max"] - x["T.Min"]))+ x["RHminT"]
                                                    if x["T.Max"] != x["T.Min"] else x["RHmaxT"], axis = 1)

    
    

    columns = ["Size", "Temperature", "Irradiance", "Humidity", "Yield"]
    columns.extend(hours)
    columns.extend(months)                                            
    new_data = all_data[columns]

    df = pd.concat([df, new_data], ignore_index=True)
    return df

    

if __name__ == "__main__":

    solar_sites_sizes = {
                'AliceSprings1' : 10.4, 'AliceSprings2' : 10.1, 'AngleVale' : 5.25, 'AnnaBay' : 5.04, 'Ashdale' : 10.4, 'Barnsley' : 5.04, 'Blacktown' : 20.46, 
                'Broadmeadow' : 3.15, 'Canberra' : 34.79, 'Canterbury' : 3.06, 'Carlton' : 5.1, 'Chinchilla' : 25, 'EdiUpper' : 5.3, 'ElizabethGrove' : 11.9, 
                'ElizabethNorth' : 8.32, 'Goodwood' : 12, 'Hamilton' : 10.4, 'Katherine' : 10.4, 'Kersbrook' : 12.6, 'KillarneyHeights' : 5.04, 'Koroit' : 2.04, 
                'Launceston' : 17.5, 'Mackay' : 15.5, 'Maffra' : 5.3, 'Manangatang' : 12.24, 'Merriwa' : 12.5, 'Midwest' : 12.5, 'Moonah' : 6.3, 'Newtown' : 5.1, 
                'Nowra' : 5.1, 'OceanReef' : 3.15, 'Paracomb' : 12, 'ParaVista' : 14, 'ReedyCreek' : 4.56, 'Rutherford' : 2.04, 'Serpentine' : 11.97, 
                'Sutherland' : 5.04, 'TheGap' : 5.44, 'Toowoomba' : 6.3, 'Warrnambool' : 30.6, 'WestHobart' : 9, 'Yolla' : 16
    }

    hours = ["H" + str(i) for i in range(0, 24)]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    columns = ["Size", "Temperature", "Irradiance", "Humidity"]
    columns.extend(hours)
    columns.extend(months)
    columns.append("Yield")

    df = pd.DataFrame(columns = columns)
    
    for i in solar_sites_sizes:
        df = frame_filler(df, i, solar_sites_sizes[i])


    size_max = df["Size"].max()
    size_min = df["Size"].min()
    yield_max = df["Yield"].max()
    yield_min = df["Yield"].min()
    temp_max = df["Temperature"].max()
    temp_min = df["Temperature"].min()
    irr_max = df["Irradiance"].max()
    irr_min = df["Irradiance"].min()
    hum_max = df["Humidity"].max()
    hum_min = df["Humidity"].min()


    df["Size"] = df["Size"].apply(lambda x: (x - size_min) / (size_max - size_min))
    df["Yield"] = df["Yield"].apply(lambda x: (x - yield_min) / (yield_max - yield_min))
    df["Temperature"] = df["Temperature"].apply(lambda x: (x - temp_min) / (temp_max - temp_min))
    df["Irradiance"] = df["Irradiance"].apply(lambda x: (x - irr_min) / (irr_max - irr_min))
    df["Humidity"] = df["Humidity"].apply(lambda x: (x - hum_min) / (hum_max - hum_min))

    df = df.sample(frac = 1)
    df.to_csv("FinalData.csv", index = False, header=False, sep = " ")

