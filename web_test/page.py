from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from locator import *
from element import BasePageElement

class SearchTextElement(BasePageElement):
    locator = "rawtext"

class QueryElement(BasePageElement):
    locator = "user_question"

class BasePage(object):

    def __init__(self, driver):
        self.driver = driver

# use the methods from BasePage
class MainPage(BasePage):

    # everytime we call search_text_element,
    search_text_element = SearchTextElement()
    query_element = QueryElement()


    # check if text in the title
    def is_title_matches(self):
        # returns a Boolean statement
        return "Dynamic Data Statements" in self.driver.title

    # def click_go_button(self):
    #     # * means unpack *(1,2)((one object)) -> 1, 2(two objects)
    #     WebDriverWait(self.driver, 10).until(
    #         EC.element_to_be_clickable((By.ID, "test_submit"))
    #     ).click()
    #     #element = self.driver.find_element(*MainPageLocator.GO_BUTTON)
    #     #element.click()

    def click_go_button(self):
        # * means unpack *(1,2)((one object)) -> 1, 2(two objects)
        element = self.driver.find_element(*MainPageLocator.GO_BUTTON)
        element.click()

    def click_download_button(self):
        element = self.driver.find_element_by_id("download_dataframe")
        element.click()

    def click_upload_button(self, file_path):
        file = self.driver.find_element_by_name("complete_file")
        file.send_keys(
            file_path)
        element = self.driver.find_element(*MainPageLocator.SUBMIT_BUTTON)
        element.click()

    def click_analysis_button(self):
        # * means unpack *(1,2)((one object)) -> 1, 2(two objects)
        element = self.driver.find_element(*MainPageLocator.ANALYSIS_BUTTON)
        element.click()

class SearchResultPage(BasePage):

    def success_file_process(self):
        return "Your file is ready for download!" in self.driver.page_source

    def success_file_upload(self):
        return "Bar Graphs" in self.driver.page_source

    def click_analysis_button(self):
        # * means unpack *(1,2)((one object)) -> 1, 2(two objects)
        element = self.driver.find_element(*MainPageLocator.ANALYSIS_BUTTON)
        element.click()

    def click_query_button(self):
        element = self.driver.find_element(*MainPageLocator.QUERY_BUTTON)
        element.click()

    def success_dataframe_display(self):
        return "Female Professions" in self.driver.page_source

    def success_specific_dataframe(self):
        return "Nouns" in self.driver.page_source
