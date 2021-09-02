# anyway we locate elements, we keep it in one piece

from selenium.webdriver.common.by import By

# define a class that defines all buttons on the main page
class MainPageLocator(object):
    GO_BUTTON = (By.ID, "test_submit")
    SUBMIT_BUTTON = (By.ID, "test_upload")
    ANALYSIS_BUTTON = (By.ID, "analysis_link")
    QUERY_BUTTON = (By.ID, "query_submit")
    DEBIAS_BUTTON = (By.ID, "debias_link")
    THRESHOLD_BUTTON = (By.ID, "threshold_submit")

# this will be added later on
class SearchResultsPageLocators(object):
    pass