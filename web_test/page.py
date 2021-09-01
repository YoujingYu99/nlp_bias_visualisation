from locator import *
from element import BasePageElement

class SearchTextElement(BasePageElement):
    locator = "rawtext"

# class GoButtonElement(BasePageElement):
#     locator = "go"

file_path = "C://Users//Youjing Yu//PycharmProjects//visualising_data_bias//web_test//complete_file (32).xlsx"
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
        return "Dynamic Data Statements" in self.driver.title

    def click_go_button(self):
        # * means unpack *(1,2)((one object)) -> 1, 2(two objects)
        element = self.driver.find_element(*MainPageLocator.GO_BUTTON)
        element.click()

    def click_download_button(self):
        element = self.driver.find_element_by_link_ID("download_dataframe")
        element.click()

    def click_upload_button(self, file_path):
        file = self.driver.find_element_by_name("complete_path")
        file.send_keys(
            file_path)
        element = self.driver.find_element_by_ID(*MainPageLocator.SUBMIT_BUTTON)
        element.click()

class SearchResultPage(BasePage):

    def success_file_process(self):
        return "Your file is ready for download!" in self.driver.page_source

    def success_file_upload(self):
        return "Bar Graphs" in self.driver.page_source

