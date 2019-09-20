# Code used in "Automated tracking of level of consciousness and delirium in critical illness using deep learning"

The architecture of this folder is

* step1_process_each_file.py: generate the preprocessed EEG for each recording
* step2_generate_Xy_hdf5.py: generate a single hdf5 file to contain all the preprocessed EEG
* read_delirium_data.py: load the dataset (such as the ones in example_data)
* segment_EEG.py: EEG preprocessing
* peakdetect.py: a helper function in EEG preprocessing
* RASS_prediction
    * step3_predict_RASS.py: train the CNN-LSTM model for tracking RASS
    * step4_predict_RASS_baselinemethods.py: train the baseline models for tracking RASS
    * step4_features_rass.pickle: the features for the baseline models generated from step4, for tracking RASS
    * step4_features_camicu.pickle: the features for the baseline models generated from step4, for tracking CAM-ICU
    * RASS_folds_info_10-fold.pickle: the folds for cross-validation
    * results
        * results_RASS_fold [X].pickle: contains results from the X-th fold
    * results_spect
        * results_RASS_fold [X].pickle: contains results from the X-th fold
    * results_bandpower
        * results_RASS_fold [X].pickle: contains results from the X-th fold
    * results_baselinemodel
        * results_rass.pickle: contains results from using baseline methods for tracking RASS
        * results_camicu.pickle: contains results from using baseline methods for tracking CAM-ICU
    * myml_lib
        * dataset.py: The mini-batch generator for training the model
        * experiment.py: The training logics, such as mini-batch gradient descent
        * mymodels
            * eegnet_feedforward.py: The CNN part of the model
            * eegnet_recurrent.py: The LSTM part of the model
            * ordistic_regression.py: The ordistic loss
    * models
        * model_RASS_cnn_fold [X].pth: the trained CNN from the X-th fold
        * model_RASS_rnn1_fold [X].pth: the trained first layer of LSTM from the X-th fold
        * model_RASS_rnn2_fold [X].pth: the trained second layer of LSTM from the X-th fold
* CAMICU_prediction
    * step3_predict_CAMICU.py: train the CNN-LSTM model for tracking CAM-ICU
    * results
        * results_CAMICU_fold [X]_nfix [Y].pickle: contains results from the X-th fold when fixing the first [Y] layers in the RASS model
    * myml_lib
        * dataset.py: The mini-batch generator for training the model
        * experiment.py: The training logics, such as mini-batch gradient descent
        * mymodels
            * eegnet_feedforward.py: The CNN part of the model
            * eegnet_recurrent.py: The LSTM part of the model
    * models
        * model_CAMICU_cnn_fold [X]_nfix [Y].pth: the trained CNN from the X-th fold when fixing the first [Y] layers in the RASS model
        * model_CAMICU_rnn1_fold [X]_nfix [Y].pth: the trained first layer of LSTM from the X-th fold when fixing the first [Y] layers in the RASS model
        * model_CAMICU_rnn2_fold [X]_nfix [Y].pth: the trained second layer of LSTM from the X-th fold when fixing the first [Y] layers in the RASS model
* figures
    * figure[X]_...py/.png/.pdf: the code and generated figures in png and pdf formats
    * estimate_IRR.py: estimate the inter-rater reliability
* data
    * data_list.txt: the list of paths of all files
    * err_subject_reason.txt: the list of recordings which had errors during loading/preprocessing (step1)
