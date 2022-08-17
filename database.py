import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle as pl
import os 


class surveys:

    def __init__(self, database='/global/u2/l/lonappan/workspace/LBlens/surveys.db'):

        self.database = database
        self.engine = create_engine(f'sqlite:///{self.database}', echo=False)
        self.tables = self.engine.table_names()


    def get_table_dataframe(self,table):
        if table not in self.tables:
            raise ValueError(f"{table} not in {self.tables}")
        connection = self.engine.connect()
        df = pd.read_sql_table(table,connection)
        connection.close()
        return df
    
    def write_table_dic(self,dic,table):
        df = pd.DataFrame.from_dict(dic)
        connection = self.engine.connect()
        df.to_sql(table,connection)
        connection.close()
        
    def write_table_df(self,df,table):
        connection = self.engine.connect()
        df.to_sql(table,connection)
        connection.close()

def arc2cl(arc):
    return np.radians(arc/60)**2
def cl2arc(cl):
    return np.rad2deg(np.sqrt(cl))*60

def noise(arr):
    return cl2arc(1/sum(1/arc2cl(arr)))


class Surveys:

    def __init__(self,database='surveys.pkl'):
        dirpath = os.path.dirname(os.path.realpath(__file__))
        database = os.path.join(dirpath,database)
        self.database = pl.load(open(database,'rb'))
        self.tables = self.database.keys()
        print(f'DATABASE INFO: File - {database}')

    def get_table_dataframe(self,table):
        if table not in self.tables:
            raise ValueError(f"{table} not in {self.tables}")
        return pd.DataFrame.from_dict(self.database[table])

