# Task 1



# Task 2 - Data Cleaning
Initially we planned on using the smoking & Alcohol dataset for each task. However due to it not having any missing data we will be swapping between datasets to exemplify the task the best.
For this task we used the energy dataset where we had already seen that there were missing values, and we identified this by running:
![[missing_value_code.png]]
This grants us a view telling us that there are 18-19 columns of missing data as well as some columns which contain null values. Since running .dropna on this dataset would result in no data being kept, we instead add a couple of separate pruning steps. First we add a threshold checking if a row has more than 80% missing values, in which case we drop the whole row. Then we check for any fully NaN column, or any columns with only 0 as the data values. We drop these columns as well as they do not provide any valuable data as they have a constant value. 

The reason we decided to delete missing rows and columns is due to the minimal impact it has on the dataset. The dataset we have chosen is huge and for that reason, after removing the columns with no valuable data there were only 18 rows to delete. This equates to 0.05% of the entire dataset. We believe this wont have a substantial impact on any results we gain from this dataset.
Below is an overview of the medians in the dataset before culling. Where empty columns have a red marker next to them.
![[empty_columns.png]]
Below is the medians after cleaning:
![[median_values.png]]
The keen-eyed-observer will see that "generation marine" is still there even if its a 0 value column. We are not sure why this has happened, as the column contains no outliers and is fully composed of the value 0.0 . Since its not the end of the world to keep one column of no substantial value we haven't spent more time trying to identify why it is still there. 