import datetime
import pickle
import numpy as np
import pandas as pd


if __name__=='__main__':
    
    df = pd.read_csv('data/rass_times.txt', sep='\t')
    df_Time = np.array([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in df.Time])
    
    staffs = np.sort(np.unique(df.Staff))
    researchers = sorted(set(staffs)-set([0]))
    nurse_mask = df.Staff==0
    nurse_ids = np.where(nurse_mask)[0]
    all_patients = df.PatientID.values
    
    ynurses = {}
    yresearchers = {}
    time_diffs = {}
    patients = {}
    for researcher in researchers:
        
        staff_ids = np.where(df.Staff==researcher)[0]
        ynurse = []
        yresearcher = []
        time_diff = []
        patient = []
        for i in staff_ids:
            this_ids = np.where(nurse_mask&(all_patients==df.PatientID[i]))[0]
            if len(this_ids)==0:
                continue
                
            dt = np.abs(df_Time[this_ids]-df_Time[i])
            closest_nurse_id = this_ids[np.argmin(dt)]
            this_time_diff = np.min(dt).total_seconds()
            if this_time_diff>=4*3600:
                continue
                
            ynurse.append(df.Score[closest_nurse_id])
            yresearcher.append(df.Score[i])
            time_diff.append(this_time_diff)
            patient.append(df.PatientID[closest_nurse_id])
            
        if len(ynurse)<10:
            continue
        ynurses[researcher] = np.array(ynurse)
        yresearchers[researcher] = np.array(yresearcher)
        time_diffs[researcher] = np.array(time_diff)
        patients[researcher] = np.array(patient)
    
    researchers = sorted(ynurses.keys())
    with open('IRR.pickle', 'wb') as ff:
        pickle.dump([ynurses, yresearchers, time_diffs, patients, researchers], ff, protocol=2)
               
