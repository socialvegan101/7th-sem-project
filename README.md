# stock prediction
this project is intended to provide the real time prediction of the stock price(close) through the use of linear regression model. Also a separate model with RELU activation function has been prepared to utilize the LSTM feature in the data prediction. 
 

## Data
The data can be found in [data/company] folder and are arranged according to the company. For example: in `NMB.csv` you can find all the data of NMB Bank Limited sorted in ascending order (date-wise).

The repository currently includes data of around 100 companies.


## Code
The code that is used in prediction resides on the [src/] folder. The code is written in `python3.14`.

The code that updates data on a daily basis resides on the `.github/.workflows` directory and runs on Github Action as a CRON job. The Github Action workflow runs every day (Sunday to Friday) to collect the daily data.
