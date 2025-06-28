To **review the specific changes**, refer to the commit history:  
[**View commit differences here**](https://github.com/eliasbenussi/nn-cert-individual-fairness/commit/e4342b56dbac7c3b249e8f7980d5e32fc137b44e)  


Also, note that Python 3.8 is probably needed.

### To run the original script
```
python certification_experiments.py
```

### A clean training
```python
from training import train_individually_fair_subspace_robust_model

config = {
    'dataset_name': 'crime',
    'sensitive_features': [
        'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'
    ],
    'drop_columns': [
        'HispPerCap', 'LandArea', 'LemasPctOfficDrugUn', 'MalePctNevMarr', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'MedRent', 'MedYrHousBuilt', 'OwnOccHiQuart', 'OwnOccLowQuart', 'OwnOccMedVal', 'PctBornSameState', 'PctEmplManu', 'PctEmplProfServ', 'PctEmploy', 'PctForeignBorn', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRecent', 'PctRecImmig10', 'PctRecImmig5', 'PctRecImmig8', 'PctRecentImmig', 'PctSameCity85', 'PctSameState85', 'PctSpeakEnglOnly', 'PctUsePubTrans', 'PctVacMore6Mos', 'PctWorkMom', 'PctWorkMomYoungKids', 'PersPerFam', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous', 'RentHighQ', 'RentLowQ', 'Unnamed: 0', 'agePct12t21', 'agePct65up', 'householdsize', 'indianPerCap', 'pctUrban', 'pctWFarmSelf', 'pctWRetire', 'pctWSocSec', 'pctWWage', 'whitePerCap'
    ],
    'training_goal': 'regression',
    'vanilla_epochs': 35,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.02,
    'fair_epochs': 100,
    'fair_batch_size': 1500,
    'reg': 0.02,
    'n_units': [8],
    'lr': 0.001,
    'debiased_training': False,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode':
    'milp',
    'training_verif_time_limit': None
}

_, _, model = train_individually_fair_subspace_robust_model(a_list_of_sensitive_features, a_list_of_sens_cat_cols, X_df, y_df, test_X_df, test_y_df, a_list_of_sensitive_dfs, config, save_directory=save_directory)

print(model.summary())
```
