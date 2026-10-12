During development, I focused first on building a working end-to-end
pipeline: downloading public data, validating the required columns,
creating a daily time series, evaluating forecasts, and saving outputs.
I compared two transparent baselines rather than introducing a complex
model before establishing a benchmark. I also clarified the forecasting
target: regional alert-start records per UTC day, not unique nationwide
alerts or attacks. A key limitation is that missing dates are filled
with zero, although the source may not guarantee complete reporting for
every day. The final version uses rolling one-day-ahead evaluation on
the latest 28 complete days, excludes the current incomplete UTC day,
and documents the data-completeness assumption. The results should be
interpreted as a baseline experiment, not an operational warning system.
