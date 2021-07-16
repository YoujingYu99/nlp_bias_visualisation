"""
This file is to translate the natural language input at the web interface 
to query language that can be used to extract information from the corpora
"""
import os

nl_query = "Are most nurses women ?"  # pre-set, should be attained from the webpage input
database = "amalgum"  # should be attained from the webpage input

"""need to be redefined in later stages"""
path = "amalgum/amalgum/"

"""to understand the user input"""
"""consider using the in2sql package"""
"""however not sure if we can transform the database into sql database"""
"""can look into the code of other data visualisation site and see what they did in the code"""


class UserParser:
    def __init__(self):
        self.input = None
        self.state = False  # True for success, False otherwise
        self.flag = 0
        """
        0: SQL
        1: noSQL
        """

    def ln2sql(self):
        """to be finished later"""
        return # or should generate a json file for the downstream






