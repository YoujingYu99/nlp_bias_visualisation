from locator import *
from element import BasePageElement

class SearchTextElement(BasePageElement):
    locator = "q"

# class GoButtonElement(BasePageElement):
#     locator = "go"


class BasePage(object):

    def __init__(self, driver):
        self.driver = driver

# use the methods from BasePage
class MainPage(BasePage):

    # everytime we call search_text_element,
    search_text_element = SearchTextElement()

    # check if text in the title
    def is_title_matches(self):
        # returns a Boolean statement
        return "Python" in self.driver.title

    def click_go_button(self):
        # * means unpack *(1,2)((one object)) -> 1, 2(two objects)
        element = self.driver.find_element(*MainPageLocator.GO_BUTTON)
        element.click()

class SearchResultPage(BasePage):

    def is_results_found(self):
        return "No results found." not in self.driver.page_source

