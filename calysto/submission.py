import os
import unittest
import pandas as pd
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleSubmission:

    def __init__(self, kaggle_params):
        self.kaggle_params = kaggle_params

# class KaggleSubmission(DataFrame):

#     @staticmethod
#     def _validateColumns(self):
#         if self.target_column not in self.columns or self.target_column not in self.columns:
#             raise AttributeError("O dataframe de submissão não possui as colunas requeridas")

#     @staticmethod
#     def _validateFolder(self):
#         if not(os.path.exists(self.folder_name)):
#             raise AttributeError("O diretório informado não existe")

#     @property
#     def _constructor(self):
#         return KaggleSubmission

    # @property
    def save(self, test, prediction, folder_name = "submissions"):
        self.test = test
        self.prediction = prediction
        self.competition_name = self.kaggle_params["competition_name"]
        self.id_column = self.kaggle_params["id_column"]
        self.target_column = self.kaggle_params["target_column"]
        self.folder_name = folder_name

        # self._validateColumns(self)
        # self._validateFolder(self)

        df = pd.DataFrame()
        df[self.id_column] = self.test[self.id_column]
        df[self.target_column] = self.prediction

        self.now = datetime.now().strftime("%Y-%b-%d-%H-%M-%S")

        submission_filename = "{}-{}.{}".format(self.competition_name, self.now, 'csv')

        self.submission_pathname = "{}/{}".format(folder_name, submission_filename)

        df.to_csv(self.submission_pathname, index = False)
        #self.to_csv(self.submission_pathname, index = False)

        return self.submission_pathname

    @property
    def submit(self):

        # Autenticação à Api do Kaggle

        api = KaggleApi()

        api.authenticate()

        api.competition_submit(file_name = self.submission_pathname, message = self.now, competition = self.competition_name)

        return "https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submissions"