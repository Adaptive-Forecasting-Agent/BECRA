# Data layout

CSV paths must match `becra/config.py` (`get_dataset()`):

```
data/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
└── weather/
    └── weather.csv
```

Download from [Time-Series-Library](https://github.com/thuml/Time-Series-Library) or standard long-term forecasting benchmarks if files are not bundled with your copy.
