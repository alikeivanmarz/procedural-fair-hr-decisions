# Phase-6 SHAP Summary


## acs_income / RF / group=all

| feature | mean_abs_shap | normalised_share | is_sensitive | is_proxy |
|---|---|---|---|---|
| WKHP | 0.110001 | 0.1966 | False | False |
| OCCP | 0.106835 | 0.1909 | False | False |
| SCHL | 0.091389 | 0.1633 | False | True |
| RELP | 0.068330 | 0.1209 | False | False |
| AGEP | 0.066883 | 0.1195 | False | False |

## acs_income / RF / group=majority

| feature | mean_abs_shap | normalised_share | is_sensitive | is_proxy |
|---|---|---|---|---|
| WKHP | 0.110452 | 0.1953 | False | False |
| OCCP | 0.107539 | 0.1901 | False | False |
| SCHL | 0.093303 | 0.1649 | False | True |
| AGEP | 0.069261 | 0.1224 | False | False |
| RELP | 0.066407 | 0.1162 | False | False |

## acs_income / RF / group=minority

| feature | mean_abs_shap | normalised_share | is_sensitive | is_proxy |
|---|---|---|---|---|
| WKHP | 0.109388 | 0.1985 | False | False |
| OCCP | 0.105878 | 0.1921 | False | False |
| SCHL | 0.088790 | 0.1611 | False | True |
| RELP | 0.070942 | 0.1274 | False | False |
| AGEP | 0.063653 | 0.1155 | False | False |

## ibm_hr_attrition / GB / group=Female

| feature | mean_abs_shap | normalised_share | is_sensitive | is_proxy |
|---|---|---|---|---|
| OverTime | 0.464908 | 0.1648 | False | False |
| StockOptionLevel | 0.312061 | 0.1106 | False | False |
| NumCompaniesWorked | 0.229742 | 0.0807 | False | False |
| Age | 0.205043 | 0.0720 | False | False |
| JobRole | 0.199537 | 0.0707 | False | False |

## ibm_hr_attrition / GB / group=Male

| feature | mean_abs_shap | normalised_share | is_sensitive | is_proxy |
|---|---|---|---|---|
| OverTime | 0.425640 | 0.1585 | False | False |
| StockOptionLevel | 0.301154 | 0.1122 | False | False |
| NumCompaniesWorked | 0.197307 | 0.0731 | False | False |
| JobRole | 0.187035 | 0.0697 | False | False |
| Age | 0.183265 | 0.0679 | False | False |

## ibm_hr_attrition / GB / group=all

| feature | mean_abs_shap | normalised_share | is_sensitive | is_proxy |
|---|---|---|---|---|
| OverTime | 0.441133 | 0.1611 | False | False |
| StockOptionLevel | 0.305457 | 0.1115 | False | False |
| NumCompaniesWorked | 0.210105 | 0.0762 | False | False |
| JobRole | 0.191968 | 0.0701 | False | False |
| Age | 0.191858 | 0.0696 | False | False |

## oulad / LR / group=all

| feature | mean_abs_shap | normalised_share | is_sensitive | is_proxy |
|---|---|---|---|---|
| highest_education | 0.033760 | 0.2199 | False | False |
| imd_band | 0.029687 | 0.1934 | False | False |
| num_of_prev_attempts | 0.021801 | 0.1420 | False | False |
| age_band | 0.021236 | 0.1383 | False | False |
| code_presentation | 0.077797 | 0.1260 | False | False |
