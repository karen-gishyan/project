<!-- ## Data Generation

1. Use `PopulateModels` from `mimic3/scripts.py` to load data into postgres.
2. Use the queries in `mimic3/sql/main.sql` to generate features and drugs, which are stored in `datasets/sqldata/`.
3. Use `mimic3/data.py` for creating data with stages for each diagnosis. The results are stored in `datasets/` under each diagnosis name. -->

First install requirements.txt in a virtual environment.

## MultiStage iterative classification experiment
To run the experiment:
```
cd mimic3/multistage
python main.py
```

## Multistage drug-treatment generation experiment

To run the experiment:
```
cd mimic3/cluster
python main.py
```

## Multistage graph-construction and search experiment

To run the experiment:

```
cd mimic3/search
python main.py
```
To test lambda pipeline on your local server:

```
python manage.py runserver
```
then send a post request to `/upload-features/`, check the response, follow the url for DAG visualization.


