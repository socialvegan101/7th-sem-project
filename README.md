# nepse-data

nepse-data is the collection of all the datasets (from past to present) of various companies listed in the Nepal Stock Market. This repository is built for the students and people who want to perform analysis on Nepal Stock Market using historical (& current) data.

## Data

The data can be found in [data/company] folder and are arranged according to the company. For example: in `NMB.csv` you can find all the data of NMB Bank Limited sorted in ascending order (date-wise).

The repository currently includes data of around 130 companies.

The Github Actions updates the data on an almost daily basis so that the datasets available here are up to date.

## Code

The code through which the data were/are being collected resides on the [src/] folder. The code is written in `python3.14` and the required library is stored in `src/requirements.txt`


The code that updates data on a daily basis resides on the `.github/.workflows` directory and runs on Github Action as a CRON job. The Github Action workflow runs 5 times every day (Sunday to Friday) so that if the data collection is missed first time then it will work the second time and so on.
