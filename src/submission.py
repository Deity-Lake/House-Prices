import os
import unittest
#import pandas as pd
from pandas import DataFrame
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
 
# class KaggleCompetition():

#     def __init__(self, competition_name, id_column, target_column):
#         self.competition_name = competition_name
#         self.id_column = id_column
#         self.target_column = target_column

class KaggleSubmission(DataFrame):

    @staticmethod
    def _validateColumns(self):
        if self.target_column not in self.columns or self.target_column not in self.columns:
            raise AttributeError("O dataframe de submissão não possui as colunas requeridas")

    @staticmethod
    def _validateFolder(self):
        if not(os.path.exists(self.folder_name)):
            raise AttributeError("O diretório informado não existe")

    @property
    def _constructor(self):
        return KaggleSubmission

    # @property
    def save(self, competition_name, id_column, target_column, folder_name = "submissions"):

        self.competition_name = competition_name
        self.id_column = id_column
        self.target_column = target_column
        self.folder_name = folder_name

        self._validateColumns(self)
        self._validateFolder(self)

        self.now = datetime.now().strftime("%Y-%b-%d-%H-%M-%S")

        submission_filename = "{}-{}.{}".format(self.competition_name, self.now, 'csv')

        self.submission_pathname = "{}/{}".format(folder_name, submission_filename)

        self.to_csv(self.submission_pathname, index = False)

        return self.submission_pathname

    @property
    def submit(self):

        # Autenticação à Api do Kaggle

        api = KaggleApi()

        api.authenticate()

        api.competition_submit(file_name = self.submission_pathname, message = self.now, competition = self.competition_name)