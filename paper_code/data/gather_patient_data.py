import pdb
import os
import os.path
import pickle
import sys
import numpy as np
import scipy.io as sio
import hdf5storage as hs
import pandas as pd
import mne
sys.path.insert(0, r'/data/delirium/mycode_waveform_ordinal')
from read_delirium_data import datenum, SECONDS_IN_DAY


drug_table_path = '/data/delirium/delirium_database_tables/vInjected_Drug.csv'
main_record_path = '/data/delirium/delirium_database_tables/Main_Record.csv'
icused_record_path = '/data/delirium/delirium_database_tables/ICU_Sed_Record.csv'
icused_assessment_path = '/data/delirium/delirium_database_tables/ICU_Sed_Assessment.csv'
icused_charlson_comorbidity_path = '/data/delirium/delirium_database_tables/vICU_Sed_Charlson_Comorbidity.csv'
time_value_path = '/data/delirium/delirium_database_tables/Time_Value.csv'
patient_path = '/data/delirium/delirium_database_tables/Patient.csv'
or_record_path = '/data/delirium/delirium_database_tables/vOR_Record.csv'
race_LUT_path = '/data/delirium/delirium_database_tables/Race_Text.csv'
misc_parameter_LUT_path = '/data/delirium/delirium_database_tables/Misc_Parameter.csv'
unit_lut_path = '/data/delirium/delirium_database_tables/Unit_Text.csv'
event_path = '/data/delirium/delirium_database_tables/Events.csv'
feature_root_path = '/data/delirium/features_all'

drug_tostudy = np.sort(['Dexmedetomidine', 'Hydromorphone', 'Propofol', 'Midazolam',
 'Fentanyl', 'Labetalol', 'Lidocaine', 'Ketamine',
 'Epinephrine', 'Haloperidol', 'Morphine',
 'Esmolol',])
# 'Norepinephrine', 'Phenylephrine' # blood pressure
# 'Cisatracurium', 'Pancuronium'  # muscle relaxant

"""
# read drug information from ICU_Sed_eMAR_Drug
drug_path = r'/data/delirium/delirium_database_tables/vICU_Sed_eMAR_Drug.csv'
drug_data = pd.read_csv(drug_path, sep=',')
drug_data = drug_data.assign(TimeStamp=pd.Series(map(lambda x:datenum(x, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True), drug_data.Time)).values)  # add a column TimeStamp = datenum(Time)
assert drug_data.Status[drug_data.Is_Infusion==0].unique().tolist()==['Given']
print(drug_data.Status.unique())
print(drug_data.Drug.unique())

drug_trace = {}
unique_patients = drug_data.PatientID.unique()
for patientid in unique_patients:
    print(patientid)
    ids_p = np.where(drug_data.PatientID==patientid)[0]
    unique_drugs = drug_data.Drug[ids_p].unique()
    
    drug_trace[patientid] = {}
    for drug in unique_drugs:
        ids_pd = ids_p[drug_data.Drug[ids_p]==drug]
        ids_pd = ids_pd[np.argsort(drug_data.TimeStamp[ids_pd].values)] # sort timestamp in ascend order
        drug_trace[patientid][drug] = drug_data.iloc[ids_pd]
        
with open('drug_data_ICU_Sed_eMAR_Drug.pickle', 'wb') as f:
    pickle.dump(drug_trace, f, protocol=2)
"""
def get_change_points(notnan_mask, padding=0, end_plus_1=True):
    change_points = np.r_[np.where(np.roll(notnan_mask,1) != notnan_mask)[0], len(notnan_mask)]
    change_points = np.sort(np.unique(change_points))
    change_points = change_points.repeat(2)[1:-1].reshape(-1,2)
    change_points[:,0] -= padding  # expand change points to pad
    change_points[:,1] += padding
    change_points = np.clip(change_points, 0, len(notnan_mask))
    if not end_plus_1:
        change_points[:,1] -= 1
    
    return change_points

def generate_MRN2PatientID(RecordID2PatientID=None):
    # convert MRN -(Main_Record)-> RecordID -(ICU_Sed_Record)->PatientID
    MainRecord = pd.read_csv(main_record_path, sep=',')
    unique_mrns = MainRecord.MRN.unique()
    MRN2RecordID = {mrn: MainRecord.RecordID[MainRecord.MRN==mrn].values for mrn in unique_mrns}
    
    if RecordID2PatientID is None:
        RecordID2PatientID = generate_RecordID2PatientID()
    
    MRN2PatientID = {}
    for mrn in unique_mrns:
        rids = MRN2RecordID[mrn]
        pids = []
        for rr in rids:
            if rr in RecordID2PatientID:
                pids.append(RecordID2PatientID[rr])
        assert len(pids)<=1 # 1 to 1
        if len(pids)==1:
            MRN2PatientID[mrn] = pids[0]
    return MRN2PatientID

def generate_RecordID2PatientID():
    ICU_Sed_Record = pd.read_csv(icused_record_path, sep=',')
    unique_rids = ICU_Sed_Record.RecordID.unique()
    RecordID2PatientID = {}
    for rid in unique_rids:
        patientids = ICU_Sed_Record.PatientID[ICU_Sed_Record.RecordID==rid].values
        assert len(patientids)==1 # 1 to 1
        RecordID2PatientID[rid] = patientids[0]
    return RecordID2PatientID

def prepare_demographics_data(or_data=None, RecordID2PatientID=None, MRN2PatientID=None, save_path=None, verbose=True):
    if RecordID2PatientID is None:
        RecordID2PatientID = generate_RecordID2PatientID()
    if MRN2PatientID is None:
        MRN2PatientID = generate_MRN2PatientID()

    # read admin/discharge assessment
    assessment = pd.read_csv(icused_assessment_path, sep=',')
    #assessment = assessment.assign(PatientID=pd.Series())
    assessment.insert(0, 'PatientID', map(lambda x:RecordID2PatientID.get(x, np.nan), assessment.RecordID))
    assessment = assessment.dropna(subset=['PatientID']).reset_index(drop=True)
    # transform into xx and past_xx
    assert len(assessment)%2==0
    assert np.all(assessment.PatientID[::2].values==assessment.PatientID[1::2].values)
    assert np.any(assessment.PatientID[::2].duplicated(keep=False))==False
    assert np.all(np.arange(len(assessment))%2==assessment.bPastHistory)
    assessment = pd.concat([assessment[::2].drop(columns=['bPastHistory','UserID']).reset_index(drop=True),
                            assessment[1::2].drop(columns=['RecordID','bPastHistory','UserID','PatientID']).reset_index(drop=True).add_prefix('past_')],
                            axis=1, ignore_index=False)
    
    # read charlson_comorbidity
    charlson_comorbidity = pd.read_csv(icused_charlson_comorbidity_path, sep=',').drop(columns='AssessedBy')
    
    #if or_data is None:
    #    # read charlson_comorbidity
    #    or_data = prepare_OR_data()
    
    # read icused_record
    icused_record = pd.read_csv(icused_record_path, sep=',').drop(columns='RecordID')
    
    # read patient
    patient_data = pd.read_csv(patient_path, sep=',')
    patient_data.insert(0, 'PatientID', map(lambda x:MRN2PatientID.get(x, np.nan), patient_data.MRN))
    patient_data = patient_data.dropna(subset=['PatientID']).reset_index(drop=True)
    patient_data = patient_data.drop(columns=['FirstName','MiddleName','LastName'])#'MRN'
    race_data = pd.read_csv(race_LUT_path, sep=',')
    id2race = {race_data.iloc[i].TextID:race_data.iloc[i].Race for i in range(len(race_data))}
    patient_data = patient_data.assign(Race=map(lambda x:id2race.get(x, np.nan), patient_data.RaceID))
    patient_data = patient_data.drop(columns='RaceID')
    
    # merge
    demographics_data = pd.merge(assessment, charlson_comorbidity, how='inner', on='PatientID')
    demographics_data = pd.merge(demographics_data, icused_record, how='inner', on='PatientID')
    demographics_data = pd.merge(demographics_data, patient_data, how='inner', on='PatientID')
    
    # sort
    demographics_data.insert(demographics_data.columns.tolist().index('PatientID')+1, 'PatientID_int', map(lambda x:int(x[len('icused'):]), demographics_data.PatientID))
    demographics_data = demographics_data.sort_values(by='PatientID_int').reset_index(drop=True)
    
    if save_path is not None:
        demographics_data.to_csv(save_path, sep=',', index=False, na_rep='NaN')
    
    return demographics_data
    
def prepare_drug_data(MRN2PatientID=None, demographics_data=None, exclude_drug=None, save_path=None, verbose=True):
    if MRN2PatientID is None:
        MRN2PatientID = generate_MRN2PatientID()

    # read drug information from vInjected_Drug
    drug_data = pd.read_csv(drug_table_path, sep=',')
    drug_data.insert(0, 'PatientID', map(lambda x:MRN2PatientID.get(x, np.nan), drug_data.MRN))
    drug_data = drug_data.dropna(subset=['PatientID']).reset_index(drop=True)
    
    # add some columns
    drug_data = drug_data.assign(StartTimeStamp=pd.Series(map(lambda x:datenum(x, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True), drug_data.StartTime)).values)  # add a column TimeStamp = datenum(Time)
    drug_data = drug_data.assign(EndTimeStamp=pd.Series(map(lambda x:datenum(x, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True), drug_data.EndTime)).values)
    
    if exclude_drug is not None:
        drug_data = drug_data[~drug_data.Drug.isin(exclude_drug)].reset_index(drop=True)
    
    # sort
    drug_data.insert(drug_data.columns.tolist().index('PatientID')+1, 'PatientID_int', map(lambda x:int(x[len('icused'):]), drug_data.PatientID))
    drug_data = drug_data.sort_values(by=['PatientID_int', 'Drug', 'StartTimeStamp']).reset_index(drop=True)
    
    # remove infeasible rows
    time_increasing_ids = np.where(drug_data.StartTimeStamp.values<drug_data.EndTimeStamp.values)[0]
    if len(time_increasing_ids)<len(drug_data):
        #print('%s %s: warning existing non-increasing drug times. Deleted.'%(patientid, drug))
        drug_data = drug_data.iloc[time_increasing_ids].reset_index(drop=True)
                
    time_nonoverlap_ids = np.where((drug_data.StartTimeStamp[1:].values>=drug_data.EndTimeStamp[:-1].values) |
                                (drug_data.PatientID[1:].values!=drug_data.PatientID[:-1].values) |
                                (drug_data.Drug[1:].values!=drug_data.Drug[:-1].values))[0]
    if len(time_nonoverlap_ids)<len(drug_data)-1:
        #print('%s %s: warning overlapping drug times found. Using the later one.'%(patientid, drug))
        drug_data = drug_data.iloc[time_nonoverlap_ids].reset_index(drop=True)
    
    if save_path is not None:
        drug_data.to_csv(save_path, sep=',', index=False, na_rep='NaN')
    
    if verbose:
        print(drug_data.Drug.unique())
    return drug_data
    """
    drug_trace = {}
    unique_patients = drug_data.PatientID.unique()
    for patientid in unique_patients:
        print(patientid)
        ids_p = np.where(drug_data.PatientID==patientid)[0]
        unique_drugs = drug_data.Drug[ids_p].unique()
        
        drug_trace[patientid] = {}
        for drug in unique_drugs:
            ids_pd = ids_p[drug_data.Drug[ids_p]==drug]
            ids_pd = ids_pd[np.argsort(drug_data.StartTimeStamp[ids_pd].values)] # sort timestamp in ascend order
            drug_trace[patientid][drug] = drug_data.iloc[ids_pd]
    with open('drug_data_Injected_Drug.pickle', 'wb') as f:
        pickle.dump(drug_trace, f, protocol=2)
    """
    
def prepare_height_weight_data(RecordID2PatientID=None, weight_save_path=None, height_save_path=None, verbose=True):
    if RecordID2PatientID is None:
        RecordID2PatientID = generate_RecordID2PatientID()
        
    misc_parameter_lut = pd.read_csv(misc_parameter_LUT_path, sep=',')
    parameter2id = {misc_parameter_lut.iloc[i].Parameter:misc_parameter_lut.iloc[i].ParameterID for i in range(len(misc_parameter_lut))}
    height_id = parameter2id['Height']
    weight_id = parameter2id['Weight']
    
    time_values = pd.read_csv(time_value_path, sep=',')
    time_values.insert(0, 'PatientID', map(lambda x:RecordID2PatientID.get(x, np.nan), time_values.RecordID))
    time_values = time_values.dropna(subset=['PatientID']).reset_index(drop=True)
    
    unit_lut = pd.read_csv(unit_lut_path, sep=',')
    id2unit = {unit_lut.iloc[i].TextID:unit_lut.iloc[i].Unit for i in range(len(unit_lut))}
    time_values = time_values.assign(Unit=map(lambda x:id2unit.get(x, np.nan), time_values.UnitID))
    time_values = time_values.drop(columns='UnitID')
    
    weight_data = time_values[time_values.ParameterID==weight_id].reset_index(drop=True)
    height_data = time_values[time_values.ParameterID==height_id].reset_index(drop=True)
    
    # remove infeasible units
    weight_data = weight_data[weight_data.Unit.isin(['kg','lb','g'])].reset_index(drop=True)
    height_data = height_data[height_data.Unit.isin(['in','cm','m'])].reset_index(drop=True)
    
    # normalize units
    ids = np.where(weight_data.Unit=='g')[0]
    if len(ids)>0:
        weight_data.loc[ids,'Value'] = weight_data.Value[ids]/1000.
        weight_data.loc[ids,'Unit'] = 'kg'
    ids = np.where(weight_data.Unit=='lb')[0]
    if len(ids)>0:
        weight_data.loc[ids,'Value'] = weight_data.Value[ids]*0.453592
        weight_data.loc[ids,'Unit'] = 'kg'
    ids = np.where(height_data.Unit=='in')[0]
    if len(ids)>0:
        height_data.loc[ids,'Value'] = height_data.Value[ids]*2.54
        height_data.loc[ids,'Unit'] = 'cm'
    ids = np.where(height_data.Unit=='m')[0]
    if len(ids)>0:
        height_data.loc[ids,'Value'] = height_data.Value[ids]*100.
        height_data.loc[ids,'Unit'] = 'cm'
    
    # remove infeasible values
    weight_data = weight_data[(weight_data.Value>0) & (weight_data.Value<500)].reset_index(drop=True)
    height_data = height_data[(height_data.Value>0) & (height_data.Value<250)].reset_index(drop=True)
    
    # add time stamp
    weight_data = weight_data.assign(TimeStamp=pd.Series(map(lambda x:datenum(x, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True), weight_data.TimeOfValue)).values)
    height_data = height_data.assign(TimeStamp=pd.Series(map(lambda x:datenum(x, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True), height_data.TimeOfValue)).values)
    
    #sort
    weight_data.insert(weight_data.columns.tolist().index('PatientID')+1, 'PatientID_int', map(lambda x:int(x[len('icused'):]), weight_data.PatientID))
    weight_data = weight_data.sort_values(by=['PatientID_int', 'TimeStamp']).reset_index(drop=True)
    height_data.insert(height_data.columns.tolist().index('PatientID')+1, 'PatientID_int', map(lambda x:int(x[len('icused'):]), height_data.PatientID))
    height_data = height_data.sort_values(by=['PatientID_int', 'TimeStamp']).reset_index(drop=True)
    
    if weight_save_path is not None:
        weight_data.to_csv(weight_save_path, sep=',', index=False, na_rep='NaN')
    if height_save_path is not None:
        height_data.to_csv(height_save_path, sep=',', index=False, na_rep='NaN')
        
    return weight_data, height_data
    
def prepare_event_data(RecordID2PatientID=None, save_path=None, verbose=True):
    if RecordID2PatientID is None:
        RecordID2PatientID = generate_RecordID2PatientID()
        
    event_data = pd.read_csv(event_path, sep=',')
    event_data.insert(0, 'PatientID', map(lambda x:RecordID2PatientID.get(x, np.nan), event_data.RecordID))
    event_data = event_data.dropna(subset=['PatientID']).reset_index(drop=True)
    
    # not for ICU patients
    
    if save_path is not None:
        event_data.to_csv(save_path, sep=',', index=False, na_rep='NaN')
        
    return event_data
    
def prepare_OR_data(MRN2PatientID=None, save_path=None, verbose=True):
    if MRN2PatientID is None:
        MRN2PatientID = generate_MRN2PatientID()

    OR_data = pd.read_csv(or_record_path, sep=',')
    OR_data = OR_data.assign(PatientID=pd.Series(map(lambda x:MRN2PatientID.get(x, np.nan), OR_data.MRN)))
    OR_data = OR_data.dropna(subset=['PatientID']).reset_index(drop=True)
    
    # not for ICU patients
    
    if save_path is not None:
        OR_data.to_csv(save_path, sep=',', index=False, na_rep='NaN')
        
    return OR_data
    
def gather_patient_data(patientid, demographics_data, drug_data, weight_data, height_data, nan_pad=None, with_spect=True, with_eeg=False, with_bsr=False):#
    """
    Returns:
    patient_label: shape = (patient_duration,)
    patient_label_info: (patient_duration,) 'rass_time\tstaffid\tstatus'
    patient_spect: shape = (patient_duration, freq_num, channel_num)
    patient_eeg: shape = (patient_duration, channel_num, T)
    patient_no_artifact: shape = (patient_duration,)
    patient_drug: {drug: (patient_duration,), drug_unit: (patient_duration,), ...}
    patient_demo: {weight: float (kg), height: float (m), age: float (yr)...}
    patient_weight_height: shape = (2, patient_duration)?
    """
    segment_step_time = 2
    Fs = 250.
    #min_timesteps = 10*60*Fs
    records = filter(lambda x:x.startswith('feature_') and x.endswith('.mat'), os.listdir(os.path.join(feature_root_path, patientid)))
    
    # determine the start and end time for a patient, which is [t0+min(seg_start_ids)-1day, t0+max(seg_start)]
    patient_start = np.inf
    patient_end = -np.inf
    seg_masks = []
    for record in records:
        this_path = os.path.join(feature_root_path, patientid, record)
        res = hs.loadmat(this_path, variable_names=['t0','seg_start_ids','subject'])#,'seg_masks'
        seg_start_ids = res['seg_start_ids'].flatten()
        #if seg_start_ids.max()-seg_start_ids.min()<=min_timesteps:
        #    #print('%s is less than 10min, ignored.'%record)
        #    continue
        subject = res['subject'][0] if type(res['subject'])==np.ndarray else res['subject']
        assert (patientid in subject and record.replace('feature_','') in subject), this_path
        this_times = seg_start_ids.astype(float)/Fs + datenum(res['t0'][0], '%Y-%m-%d %H:%M:%S.%f', return_seconds=True)  # [second]
        patient_start = min(patient_start, this_times.min())
        patient_end = max(patient_end, this_times.max()+segment_step_time)
        #seg_masks.extend(map(lambda x:x.strip().split('_')[0], res['seg_masks']))
    if np.isinf(patient_start):
        return [], [], [], [], [], [], [], {}, []#, []
    patient_start = int(np.round(patient_start))
    patient_end = int(np.round(patient_end))
    #seg_masks = np.array(seg_masks)
            
    # patient demographics
    assert len(demographics_data.PatientID)==len(demographics_data.PatientID.unique())
    patient_demo = demographics_data.iloc[np.where(demographics_data.PatientID==patientid)[0]]
    
    # add height to demographics
    this_patient_height = height_data[height_data.PatientID==patientid]
    if len(this_patient_height)<=0:
        patient_demo = patient_demo.assign(Height=[np.nan])
    else:
        start_height = this_patient_height.iloc[np.argsort(np.abs(patient_start-this_patient_height.TimeStamp.values))[0]].Value
        end_height = this_patient_height.iloc[np.argsort(np.abs(patient_end-this_patient_height.TimeStamp.values))[0]].Value
        patient_demo = patient_demo.assign(Height=[(start_height+end_height)/2.])
    
    # add weight to demographics
    this_patient_weight = weight_data[weight_data.PatientID==patientid]
    if len(this_patient_weight)<=0:
        patient_demo = patient_demo.assign(Weight=[np.nan])
    else:
        start_weight = this_patient_weight.iloc[np.argsort(np.abs(patient_start-this_patient_weight.TimeStamp.values))[0]].Value
        end_weight = this_patient_weight.iloc[np.argsort(np.abs(patient_end-this_patient_weight.TimeStamp.values))[0]].Value
        patient_demo = patient_demo.assign(Weight=[(start_weight+end_weight)/2.])
    
    # advance patient start time by 1 day to allow previous drugs
    patient_start -= int(SECONDS_IN_DAY)
    patient_duration = patient_end-patient_start  # [second]
    patient_times = np.arange(0, patient_duration)+patient_start
    #if with_eeg:
    #    patient_times_eeg = np.arange(0, patient_end-patient_start,1./Fs)+patient_start
    #else:
    #    patient_times_eeg = None
    
    # get patient_label_xxx, patient_spect, patient_no_artifact
    patient_label_info = np.array(['']*patient_duration, dtype=object)
    patient_label = np.zeros((patient_duration,6))+np.nan  # 1s resolution #TODO decide 6
    #patient_unit_route = np.array(['']*patient_duration).astype(object)
    if with_spect:
        patient_spect = np.zeros((patient_duration,79*4))+np.nan
    else:
        patient_spect = None
    if with_eeg:
        #patient_eeg = np.zeros((int(patient_duration*Fs),4))+np.nan
        patient_eeg = np.zeros((patient_duration,4, 250),dtype='float16')+np.nan
    else:
        patient_eeg = None
    if with_bsr:
        patient_bsr = np.zeros((patient_duration,4))+np.nan
    else:
        patient_bsr = None
    patient_no_artifact = np.array(['nodata']*patient_duration, dtype=object)#np.ones(patient_duration, dtype=bool)
    for record in records:
        this_path = os.path.join(feature_root_path, patientid, record)
        res = hs.loadmat(this_path, variable_names=['seg_start_ids'])
        seg_start_ids = res['seg_start_ids'].flatten()
        #if seg_start_ids.max()-seg_start_ids.min()<=min_timesteps:
        #    continue
        variable_names = ['t0','labels','seg_masks', 'assess_times']
        if with_spect:
            variable_names.append('EEG_specs')
        if with_eeg:
            variable_names.append('EEG_segs')
        if with_bsr:
            variable_names.append('burst_suppression')
        res = hs.loadmat(this_path, variable_names=variable_names)
        labels = res['labels']#.flatten()

        assess_times = np.array(map(lambda x:x.strip(), res['assess_times']))
        """
        # remove not good rass
        if len(rass_times)>0:
            ids = filter(lambda x:len(rass_times[x].strip())>0 and not rass_times[x].strip().endswith('good'), range(len(rass_times)))
            if len(ids)>0:
                rass_times[ids] = ''
                labels[ids] = np.nan
        """
            
        this_times = seg_start_ids.astype(float)/Fs + datenum(res['t0'][0], '%Y-%m-%d %H:%M:%S.%f', return_seconds=True)-patient_start
        this_times = np.round(this_times).astype(int)
        
        #fill to 1s resolution
        #TODO this is valid for segment_step_time == 2
        this_times = np.sort(np.r_[this_times, this_times+1])
        if np.any(~np.isnan(patient_label[this_times])):
            raise SystemExit('%s: cannot assign label due to overlapping'%os.path.join(patientid,record))
        else:
            patient_label[this_times] = labels.repeat(segment_step_time, axis=0)
            
        if np.any(patient_label_info[this_times]!=''):
            raise SystemExit('%s: cannot assign assessment info due to overlapping'%os.path.join(patientid,record))
        else:
            patient_label_info[this_times] = assess_times.repeat(segment_step_time, axis=0)
        #unique_assess_times = np.unique(assess_times)
        #this_assess_times = assess_times.repeat(segment_step_time)
        #for urt in unique_assess_times:
        #    if urt.strip()=='':
        #        continue
        #    this_rass_ids = np.where(this_assess_times==urt)[0]
        #    assert np.all(np.diff(this_rass_ids)==1)
        #    patient_label_info[this_times[this_rass_ids].min():this_times[this_rass_ids].max()+1] = urt.strip()
            
        if with_spect:
            if np.any(~np.isnan(patient_spect[this_times])):
                raise SystemExit('%s: cannot assign spect due to overlapping in EEG spectrogram'%os.path.join(patientid,record))
            else:
                patient_spect[this_times] = res['EEG_specs'].transpose(0,2,1).reshape(len(res['EEG_specs']),-1).repeat(segment_step_time, axis=0)
        
        if with_bsr:
            #TODO this is valid for segment_step_time == 2
            patient_bsr[this_times] = res['burst_suppression'].repeat(segment_step_time, axis=0)
                
        if with_eeg:
            if np.any(~np.isnan(patient_eeg[this_times,:,0])):
                raise SystemExit('%s: cannot assign EEG due to overlapping in EEG signal'%os.path.join(patientid,record))
            else:
                #if len(res['EEG_segs'])%2==0:
                #    this_eeg = res['EEG_segs'][::2]
                #else:
                #    this_eeg = res['EEG_segs'][::2][:-1]
                #this_eeg = this_eeg.transpose(0,2,1).reshape(-1,res['EEG_segs'].shape[1])
                #this_eeg_times = np.round(np.arange(this_times[0]*Fs,this_times[-1]*Fs)).astype(int)
                #if len(this_eeg_times)>len(this_eeg):
                #    this_eeg_times = this_eeg_times[:len(this_eeg)]
                #elif len(this_eeg_times)<len(this_eeg):
                #    this_eeg = this_eeg[:len(this_eeg_times)]
                #patient_eeg[this_eeg_times] = this_eeg
                patient_eeg[this_times] = res['EEG_segs'][:,:,::4].repeat(segment_step_time, axis=0)
                
        patient_no_artifact[this_times] = np.array(map(lambda x:x.strip().split('_')[0], res['seg_masks'])).repeat(segment_step_time)# in ['normal','spurious spectrum']
    
    # there are only 2 bolus for Fentanyl, remove them
    ids = np.where(~np.isnan(drug_data.InfusionRate))[0]
    assert len(drug_data)-len(ids)==2
    drug_data = drug_data.iloc[ids].reset_index(drop=True)
    
    # get patient_drug for each patient
    patient_drug = {}
    this_patient_drugs = drug_data[drug_data.PatientID==patientid]
    for drug in drug_tostudy:
        if drug not in this_patient_drugs.Drug.values:
            #print('%s %s: no drug data'%(patientid, drug))
            continue
            
        this_patient_drug = this_patient_drugs[this_patient_drugs.Drug==drug]
        drug_start = this_patient_drug.StartTimeStamp.min()
        drug_end = this_patient_drug.EndTimeStamp.max()
    
        if patient_start-drug_end>=0 or drug_start-patient_end>=0:
            #print('%s %s: drug not in patient time range'%(patientid, drug))
            continue
            
        #unique_units = this_patient_drug.Unit.unique()
        #assert len(unique_units)==1, 'Patient %s, Drug %s: multiple units %s'(patientid, drug, unique_units)
        #unit = unique_units[0]
        
        # convert drug_data to 1d array
        #patient_unit_route
        this_patient_drug_trace = np.zeros(patient_duration)+np.nan
        this_patient_drug_unit = np.array(['']*patient_duration, dtype=object)
        cc=0
        for ndr in range(len(this_patient_drug)):
            drug_row = this_patient_drug.iloc[ndr]
            if patient_start-drug_row.EndTimeStamp>=0 or drug_row.StartTimeStamp-patient_end>=0:
                continue
            else:
                cc+=1
            this_start_time = max(0, int(np.round(drug_row.StartTimeStamp-patient_start)))
            this_end_time = min(patient_duration, int(np.round(drug_row.EndTimeStamp-patient_start)))
            #if 0<=this_start_time<this_end_time<patient_duration:
            #    cc+=1
            #else:
            #    continue
            
            if np.any(~np.isnan(this_patient_drug_trace[this_start_time:this_end_time])):
                raise SystemExit('%s: cannot assign drug due to overlapping in drugs'%os.path.join(patientid,record,drug))
            else:
                if pd.notnull(drug_row.BolusAmount):
                    this_patient_drug_trace[this_start_time:this_end_time] = drug_row.BolusAmount #boluse_pad?
                elif pd.notnull(drug_row.InfusionRate):
                    this_patient_drug_trace[this_start_time:this_end_time] = drug_row.InfusionRate
                this_patient_drug_unit[this_start_time:this_end_time] = drug_row.Unit
        if cc==0:
            #print('%s %s: drug not in patient time range'%(patientid, drug))
            continue
        this_patient_drug_trace[this_patient_drug_trace==0] = np.nan  # amount = 0 indicates no drugs
        patient_drug[drug] = this_patient_drug_trace
        patient_drug['%s_unit'%drug] = this_patient_drug_unit
        #print('%s %s: OK'%(patientid, drug))
    
    # remove nans outside padded areas
    if with_spect:
        notnan_mask = np.logical_not(np.isnan(patient_spect[:,0])).astype(int)
    elif with_eeg:
        notnan_mask = np.logical_not(np.isnan(patient_eeg[:,0,0])).astype(int)
    else:
        notnan_mask = np.r_[np.zeros(int(SECONDS_IN_DAY)), np.ones(len(patient_times)-int(SECONDS_IN_DAY))].astype(int)
    #notnan_mask = ((np.logical_not(np.isnan(patient_drug[:,0]))) | (np.logical_not(np.isnan(patient_spect[:,0])))).astype(int)

    if nan_pad is not None:
        change_points = get_change_points(notnan_mask, padding=nan_pad)
        change_points = change_points[notnan_mask[(change_points[:,0]+change_points[:,1])//2]==1]  # mark all blank areas to remove
    else:
        change_points = get_change_points(notnan_mask, padding=0)
        if notnan_mask[(change_points[0,0]+change_points[0,1])//2]==0:
            change_points = change_points[1:] # mark the first blank area to remove

    good_ids = np.zeros(patient_duration,dtype=int)
    for ii in range(len(change_points)):
        good_ids[change_points[ii,0]:change_points[ii,1]] = 1
    good_ids = np.where(good_ids==1)[0]
    
    patient_label = patient_label[good_ids]
    patient_label_info = patient_label_info[good_ids]
    if with_spect:
        patient_spect = patient_spect[good_ids]
    if with_eeg:
        patient_eeg = patient_eeg[good_ids]
    if with_bsr:
        patient_bsr = patient_bsr[good_ids]
    patient_times = patient_times[good_ids]
    patient_no_artifact = patient_no_artifact[good_ids]
    patient_drug2 = {}
    for drug in patient_drug:
        if drug.endswith('_unit'): continue
        if not np.all(np.isnan(patient_drug[drug][good_ids])):
            patient_drug2[drug] = patient_drug[drug][good_ids]
            patient_drug2['%s_unit'%drug] = patient_drug['%s_unit'%drug][good_ids]
    patient_drug = patient_drug2
    
    # add patientid to patient_label_info    
    patient_label_info = np.array(map(lambda x:patientid+'\t'+x.strip() if x.strip()!='' else '', patient_label_info))
    
    # normalize units to mg/kg/h
    weight = patient_demo.Weight.values[0]
    unique_units = ['MG/MIN', 'mcg/Min.', 'mcg/h', 'mcg/kg/Min.', 'mcg/kg/h', 'mg/h', 'mg/kg/h']#, 'mcg'
    assert np.all(np.sort(drug_data.Unit.unique())==unique_units)
    
    for drug in patient_drug:
        if drug.endswith('_unit'): continue
        this_units = patient_drug['%s_unit'%drug]
        this_drugs = patient_drug[drug]
        
        for un in unique_units:
            ids = np.where(this_units==un)[0]
            if len(ids)<=0: continue
            
            if un=='MG/MIN':
                this_drugs[ids] = this_drugs[ids]*60./weight
            elif un=='mcg/Min.':
                this_drugs[ids] = this_drugs[ids]/1000.*60./weight
            elif un=='mcg/h':
                this_drugs[ids] = this_drugs[ids]/1000./weight
            elif un=='mcg/kg/Min.':
                this_drugs[ids] = this_drugs[ids]/1000.*60.
            elif un=='mcg/kg/h':
                this_drugs[ids] = this_drugs[ids]/1000.
            elif un=='mg/h':
                this_drugs[ids] = this_drugs[ids]/weight
                
        patient_drug[drug] = this_drugs
    
    for drug in drug_tostudy:
        if '%s_unit'%drug in patient_drug:
            patient_drug.pop('%s_unit'%drug)
    patient_drug['unit'] = 'mg/kg/h'
    
    if patient_spect is not None:
        patient_spect = patient_spect.reshape(patient_spect.shape[0],4,-1)
        
    return patient_times, patient_label, patient_label_info, patient_eeg, patient_spect, patient_bsr, patient_no_artifact, patient_drug, patient_demo#, patient_weight_height, patient_times_eeg

