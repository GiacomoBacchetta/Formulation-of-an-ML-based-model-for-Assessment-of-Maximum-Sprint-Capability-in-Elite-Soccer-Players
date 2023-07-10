# Formulation of a ML-based model for the Assessment of Maximum Sprint Capability in Elite Soccer Players

# --- Import packages ---

import pandas as pd
import numpy as np
import plotly.express as px
import json
import pynumdiff
from tqdm.notebook import tqdm

import warnings

warnings.filterwarnings('ignore')

# ---


# --- Function for conversion of string time in seconds ---

def time_to_sec(timestamp):

    t = timestamp.replace(':', '.')

    try:

        hou, min, sec, cent = map(int, t.split('.'))  # Lecce-Milan
        t = hou * 60 * 60 * 100 + min * 60 * 100 + sec * 100 + cent

    except:

        min, sec, cent = map(int, t.split('.'))  # Inter-Juventus
        t = min * 60 * 100 + sec * 100 + cent
    sec = float(t/100)

    return sec

# ----


# --- Trunc function ---

def trunc(values, decs=0):

    return np.trunc(values*10**decs)/(10**decs)

# ---


# --- Function to recognize the average peak per second in a match ---

def avg_peak(data, dt):

    data = data[~np.isnan(data)]

    diff = np.diff(data)

    peaks = np.sum((diff[:-1] > 0) & (diff[1:] < 0))
    sampling_freq = 1.0 / dt  
    avg_peaks_per_second = peaks / (len(data) / sampling_freq)
    
    param = np.exp(-1.6*np.log(avg_peaks_per_second)-0.71*np.log(dt)-5.1)
    
    return param

# ---


# --- DATA EXTRACTION ---
 
def data_extraction(match):

    # Importing match data in a Pandas dataframe 
    df_match = pd.read_excel(match+'\match_data.xlsx')

    # Deleting that players who don't join in the game
    df_match = df_match[(df_match['start_time'].notna())]

    # Deleting that players who play enough minutes
    df_match = df_match[df_match['start_time'].apply(lambda x: time_to_sec(str(x) +'.00') < time_to_sec('01:05:00.00'))]

    # Deleting goalkeepers
    goalkeepers = ['Tatarusanu', 'Falcone', 'Maignan', 'Meret', 'Musso']

    # --- Creation of the players dictionary for the chosen match ---
    players = {}

    for index, row in df_match.iterrows():

        if row['last_name'] not in goalkeepers:
            
            main_key = row['trackable_object']
            dict_row = {column: row[column]
                            for column in df_match.columns 
                                if column != 'trackable_object'}
            
            players[main_key] = dict_row

    if 'Atalanta' in match or 'Napoli' in match:

        parameter = 1
        frequency = 0.2

        # Importing tracking data in a Pandas dataframe
        df_tracking = pd.read_excel(match+'\\tracking_data.xlsx')

        for id in players.keys():

            movements = []

            for row in df_tracking.values.tolist():

                if row[0] == id:

                    # Both visible and not visible frame
                    if row[3] == 0 or row[3] == 1:  # da togliere in fase definitiva

                        if row[1] == 1 and time_to_sec(str(row[2])) < time_to_sec('00:45:00.00'):

                            positions = [row[1],    # period
                                        time_to_sec(str(row[2])),   # timestamp  
                                        row[4],     # x 
                                        row[5]]     # y
                            
                        else:

                            positions = [row[1],    # period
                                        time_to_sec(str(row[2])) + time_to_sec('00:45:00.00'),  # timestamp
                                        row[4],     # x
                                        row[5]]     # y
                            
                        movements.append(positions)

            players[id].update({'tracking': np.array(movements, dtype=object)})
            
    elif 'Lecce' in match:

        parameter = 10
        frequency = 0.1
        
        path = r""

        # Importing tracking data in a Pandas dataframe
        df_tracking = pd.read_json(path)

        list_of_dicts = []

        for dict in df_tracking['body'][0]:

            try:

                dict = json.loads(dict)  
                list_of_dicts.append(dict)  

            except:

                pass  

        # Deleting the missing values in the dataframe
        df_tracking = pd.DataFrame(list_of_dicts).dropna(
            subset=['data', 'period', 'timestamp'])

        track_list = [[row['period'], row['timestamp'], data_dict] 
                for _, row in df_tracking.iterrows() 
                for data_dict in row['data']]

        for id in players.keys():

            movements = []

            for row in track_list:

                if row[2]['trackable_object'] == id:

                    if row[2]['is_visible'] is True or row[2]['is_visible'] is False:  

                        positions = [row[0],    # period
                                    time_to_sec(row[1]),    # timestamp
                                    row[2]['x'],    # x
                                    row[2]['y']]    # y
                        movements.append(positions)

            players[id].update({'tracking': np.array(movements, dtype=object)})


    # --- Match data extraction ---
    df_possesso = pd.read_excel(match + '\possession_data.xlsx')

    period_id = df_possesso['period_id']
    possession_start = df_possesso['First(possession_start)']
    possession_end = df_possesso['Last(possession_end)']

    possession_frame = np.array([])

    for i in range(len(possession_start)):

        if period_id[i] == 1:

            possession_start[i] = time_to_sec(str(possession_start[i]))
            possession_end[i] = time_to_sec(str(possession_end[i]))

        else:
            
            possession_start[i] = time_to_sec(
                str(possession_start[i])) + time_to_sec('00:45:00.00')
            possession_end[i] = time_to_sec(
                str(possession_end[i])) + time_to_sec('00:45:00.00')

        possession_frame = np.append(possession_frame, np.arange(
            possession_start[i], possession_end[i]+0.001, 0.1))

    # ---

    return players, frequency

# ---






# --- DATA WRANGLING ---
 
def data_wrangling(players_extraction, frequency_extraction, parameter):

    for player in tqdm(players_extraction):

        tracking = players_extraction[player]['tracking'].astype(np.float32)

        x_delta = np.append(np.zeros((1, 1)), np.ediff1d(tracking[:, 2]))

        y_delta = np.append(np.zeros((1, 1)), np.ediff1d(tracking[:, 3]))

        time_delta = np.append(np.zeros((1, 1)), np.ediff1d(tracking[:, 1]))

        period_delta = np.append(np.zeros((1, 1)), np.ediff1d(tracking[:, 0]))

        x_delta[np.where(period_delta == 1)[0]] = 0
        y_delta[np.where(period_delta == 1)[0]] = 0

        space_delta = np.linalg.norm((x_delta, y_delta), axis=0)

        speed = np.append(np.zeros((1, 1)), np.divide(space_delta[1:], time_delta[1:]))

        speed, acceleration = pynumdiff.finite_difference.first_order(
            speed, frequency_extraction, [parameter], options={'iterate': True})
        
        over_space_delta = np.where(space_delta > 10.3 * frequency_extraction)[0]  # 0.92   #10.3 è 37Km/h espresso in m/s
        
        filter = np.append(over_space_delta, (over_space_delta-1, over_space_delta+1,
                        over_space_delta+2, over_space_delta-2, over_space_delta+3, over_space_delta-3))
        # filter = np.union1d(filter, np.where(speed > np.percentile(speed, 95)))
        filter = np.intersect1d(filter, np.arange(0, len(tracking)))

        space_delta[filter] = np.nan
        time_delta[filter] = np.nan
        speed[filter] = np.nan
        acceleration[filter] = np.nan
        
        distance = np.nancumsum(space_delta)
        
        tracking = np.concatenate((tracking[:,:4],
                                space_delta.reshape(-1, 1),
                                time_delta.reshape(-1, 1),
                                distance.reshape(-1, 1),
                                speed.reshape(-1, 1),
                                acceleration.reshape(-1, 1)), axis=1)

        start_time = time_to_sec(str(players_extraction[player]['start_time']) + '.00')
        idx_to_delete = np.where(tracking[:, 1] <= start_time + 1)[0]
        tracking[idx_to_delete, -2:] = np.nan

        try:

            end_time = time_to_sec(str(players_extraction[player]['end_time']) + '.00')
            idx_to_delete = np.where(tracking[:, 1] > end_time)
            tracking = np.delete(tracking, idx_to_delete, axis=0)

        except:

            pass

        tracking = tracking[np.sort(
            np.unique(tracking[:, 1], return_index=True)[1])]

        # NaN values in the timestamps of no possession
        tracking[~np.in1d(trunc(tracking[:, 1], 1), trunc(
            possession_frame[:], 1))] = np.nan  
        
        if start_time != 0:

            tracking[:5, -2:] = np.nan

        extra_idx_to_delete = np.where((tracking[:,0] == 1) & (tracking[:, 1] > time_to_sec('00:45:00.00')))
        tracking[extra_idx_to_delete, -2:] = np.nan
        
        players_extraction[player]['tracking'] = tracking

    return players_extraction
# ---




# --- DATA VISUALIZATION

def data_visualization(players_wrangled, player):

    df = pd.DataFrame({
        'time': players_wrangled[player]['tracking'][:, 1],
          'speed': players_wrangled[player]['tracking'][:, -2],
            'acceleration': players_wrangled[player]['tracking'][:, -2][:, -1]
            })

    fig = px.line(title = players_wrangled[player]['last_name'])
    fig.add_scatter(x=df['time'], y=df['speed'], mode='lines', name='Speed')
    fig.add_scatter(x=df['time'], y=df['acceleration'], mode='lines', name='Acceleration')
    fig.show()

# ---