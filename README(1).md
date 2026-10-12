# Time Series Analysis of Air Raid Alerts in Ukraine

A small Python project that analyses daily counts of recorded regional
air-raid alert starts in Ukraine and compares two simple one-day-ahead
forecasting baselines.

> **Scope and safety:** This is a retrospective time-series exercise,
> not an operational warning system, an attack predictor, or a source of
> real-time safety advice. Follow official alerts and instructions.

## Research question

Can a simple baseline based on the previous day's count or the previous
seven-day average estimate the next day's number of recorded regional
alert starts?

## Data

-   **Source:** [Ukrainian air-raid sirens
    dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
-   **File:** `datasets/volunteer_data_en.csv`
-   **Fields used:** `started_at` and `region`
-   **Time zone:** UTC
-   **Target:** number of recorded regional alert-start records per UTC
    day.

The target is a count of records in a regional dataset. It is **not**
the number of unique nationwide alert episodes, attacks, impacts, or the
duration of danger.

The script downloads the latest file available at the source when it
runs. The latest event timestamp printed by the script indicates event
recency; it does not prove that the dataset is complete or that the
source has been updated successfully.

## Methods

1.  Download the CSV and check that the required columns exist.
2.  Parse event timestamps as UTC and remove records with invalid
    timestamps or missing required values.
3.  Aggregate records into daily counts.
4.  Reindex the daily series to a continuous daily calendar, filling
    dates without records with zero.
5.  Exclude the current UTC day because it may be incomplete.
6.  Evaluate the latest **28 complete UTC days** using rolling
    one-day-ahead predictions.
7.  Compare two baselines:
    -   **Previous-day baseline:** predicts the next day's count using
        the count from the previous day.
    -   **Previous-7-day mean:** predicts the next day's count using the
        arithmetic mean of the preceding seven days.
8.  Calculate mean absolute error (MAE) for each baseline and save a
    comparison plot and CSV outputs.

For every test date, only observations strictly earlier than that date
are used to create its prediction. The test period is the most recent 28
complete days available at run time. At least 60 earlier daily
observations plus the 28 test days are required.

## Important limitations

-   A date with no rows is treated as zero recorded alert starts. The
    script cannot independently distinguish a genuine zero from missing
    or incomplete source data.
-   The source's update history and day-by-day completeness are not
    independently verified by this script. Inspect the source and output
    dates before interpreting the results.
-   The data are aggregated at regional record level; one wider alert
    episode may generate multiple regional records.
-   The baselines are intentionally simple. A lower MAE on this one test
    window does not establish general predictive performance.
-   The output is not suitable for individual safety decisions or
    operational forecasting.

## Requirements

-   Python 3.10 or later
-   pandas
-   matplotlib
-   requests

Install dependencies:

``` bash
python -m pip install -r requirements.txt
```

## Run

From the project directory:

``` bash
python main.py
```

The script downloads the dataset from the source URL and writes results
to `outputs/`.

## Output files

  -----------------------------------------------------------------------
  File                                Contents
  ----------------------------------- -----------------------------------
  `test_predictions.csv`              Actual counts and both model
                                      predictions for the test period

  `metrics.csv`                       MAE for each baseline

  `forecast_evaluation.png`           Actual counts and predictions
                                      across the test period

  `next_day_forecast.csv`             Forecast for the day after the last
                                      complete UTC day
  -----------------------------------------------------------------------

The output files are generated at runtime and are not required to be
committed to GitHub. If you include them, clearly label them with the
run date because the source dataset may change.

## Reproducibility

The script records the dataset URL in its source code and prints the
number of downloaded records, invalid timestamps removed, and latest
event timestamp. For a stronger reproducibility record, preserve the
download date and the exact source-data snapshot used for the submitted
run.

## License and attribution

The dataset is maintained by its original contributors. Review the
source repository's license and attribution requirements before
redistributing the dataset itself. This project downloads the data at
runtime and does not include a copy of the dataset.
