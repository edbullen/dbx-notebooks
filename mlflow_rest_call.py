import pandas as pd
import os
import json
import requests

"""
Example - call MLflow REST API to score a data payload using an ML model served out of MLflow

"""

# Paste the Model URL Here or set it using a config file
MLFLOW_URL = ''


def get_token():
    """ Set the Databricks PAT token as a local environment variable"""
    return os.environ.get("DATABRICKS_TOKEN")


def score_model(api_url, token, dataset: pd.DataFrame):
    """ Rows of data passed in as a Pandas DataFrame is passed to the REST API for predictions"""

    headers = {'Authorization': f'Bearer {token}'}

    data_core = dataset.to_dict(orient='records')
    data_json = {"dataframe_records": data_core}  # this is for MLflow 2.0 format
    # print(json.dumps(data_json, indent=4, sort_keys=True))

    response = requests.request(method='POST', headers=headers, url=api_url, json=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()


if __name__ == '__main__':
    token = get_token()

    if len(MLFLOW_URL) == 0:
        mlflow_url = input("Enter the URL for the MLFlow REST API: ")
    else:
        mlflow_url = MLFLOW_URL

    # 3x records to get prediction score for
    payload_df = pd.DataFrame([[1.226073, -1.640026, -1.945977, 0.505798, -0.436937, -0.765406, 0.588579, -0.326782,
                                0.600085, -0.006794, -0.979189, -0.621367, -1.192707, 0.752938, 0.916720, 0.238320,
                                -0.438378, 0.065919, -0.357661, 0.654414, 0.471472, 0.358037, -0.301632, 0.794001,
                                -0.144001, 0.717299, -0.185541, 0.012658],
                               [0.857321, 4.093912, -7.423894, 7.380245, 0.973366, -2.730762, -1.496497, 0.543015,
                                -2.35119, -3.944238, 6.355078, -7.309748, 0.748451, -9.057993, -0.648945, -1.073117,
                                1.524501, 1.831364, -0.089724, 0.483303, 0.375026, 0.1454, 0.240603, -0.234649,
                                -1.004881, 0.435832, 0.618324, 0.148469],
                               [-2.986466, -0.000891, 0.605887, 0.338338, 0.685448, -1.581954, 0.504206, -0.233403,
                                0.636768, 1.010291, 0.004518, 0.044397, 0.420853, -0.931614, 1.147974, 0.483063,
                                -0.152471, -0.285666, 0.07255, -0.764274, -0.875146, -0.509849, 1.313918, 0.355065,
                                0.448552, 0.19349, 1.214588, -0.013923]
                               ],
                              columns=["pca[0]", "pca[1]", "pca[2]", "pca[3]", "pca[4]", "pca[5]", "pca[6]", "pca[7]",
                                       "pca[8]", "pca[9]", "pca[10]", "pca[11]", "pca[12]", "pca[13]", "pca[14]",
                                       "pca[15]", "pca[16]", "pca[17]", "pca[18]", "pca[19]", "pca[20]", "pca[21]",
                                       "pca[22]", "pca[23]", "pca[24]", "pca[25]", "pca[26]", "pca[27]"])

    # Call the MLFlow Model API
    score = score_model(mlflow_url, token, payload_df)
    print(score)

