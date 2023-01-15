import pandas as pd
from sqlalchemy import create_engine
import pickle as pl
import os 


class surveys:
    """
    Class to read and write to a sqlite database of
    surveys.

    Parameters
    ----------
    database : str : path to database
    """


    def __init__(self, database='/global/u2/l/lonappan/workspace/LBlens/surveys.db'):
        self.database = database
        self.engine = create_engine(f'sqlite:///{self.database}', echo=False)
        self.tables = self.engine.table_names()


    def get_table_dataframe(self,table):
        """
        Get a table from the database as a pandas dataframe

        Parameters
        ----------
        table : str : name of table
        """
        if table not in self.tables:
            raise ValueError(f"{table} not in {self.tables}")
        connection = self.engine.connect()
        df = pd.read_sql_table(table,connection)
        connection.close()
        return df
    
    def write_table_dic(self,dic,table):
        """
        Write a dictionary to a table in the database

        Parameters
        ----------
        dic : dict : dictionary to write
        """
        df = pd.DataFrame.from_dict(dic)
        connection = self.engine.connect()
        df.to_sql(table,connection)
        connection.close()
        
    def write_table_df(self,df,table):
        """
        Write a pandas dataframe to a table in the database

        Parameters
        ----------
        df : pandas dataframe : dataframe to write
        """
        connection = self.engine.connect()
        df.to_sql(table,connection)
        connection.close()


class Surveys:
    """
    Class to read and write to a sqlite database of
    surveys.

    Parameters
    ----------
    database : str : path to database
    verbose : bool : print database info
    """

    def __init__(self,database='surveys.pkl',verbose=False):
        dirpath = os.path.dirname(os.path.realpath(__file__))
        database = os.path.join(dirpath,'Data',database)
        self.database = pl.load(open(database,'rb'))
        self.tables = self.database.keys()
        if verbose:
            print(f'DATABASE INFO: File - {database}')
        print("DATABASE: loaded")

    def get_table_dataframe(self,table):
        """
        Get a table from the database as a pandas dataframe

        Parameters
        ----------
        table : str : name of table
        """
        if table not in self.tables:
            raise ValueError(f"{table} not in {self.tables}")
        return pd.DataFrame.from_dict(self.database[table])

