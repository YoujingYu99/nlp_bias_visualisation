# anyway we locate elements, we keep it in one piece

from selenium.webdriver.common.by import By

# define a class that defines all buttons on the main page
class MainPageLocator(object):
    GO_BUTTON = (By.ID, "test_submit")
    SUBMIT_BUTTON = (By.ID, "test_upload")

# this will be added later on
class SearchResultsPageLocators(object):
    pass