"""
This file is to translate the natural language input at the web interface 
to query language that can be used to extract information from the corpora
"""
import os

nl_query = "Are most nurses women ?" # pre-set, should be attained from the webpage input
database = "amalgum" # should be attained from the webpage input

"""need to be redefined in later stages"""
path = os.path.join(database,database,'data','wikinews','AA')


