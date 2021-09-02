import unittest
from selenium import webdriver
import time
import page

test_text = "Women writers support male fighters. Male cleaners are not more careful. Lucy likes female dramas. Women do not like gloves. Lucy eats a tasty black bread. The elegant powerful woman wears shiny black glasses. The dark tall man drinks water. He adores vulnerable strong women. The kind beautiful girl picks a cup. Most writers are female. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are victims. Men are not minority. The woman is a teacher. Sarah is an engineer. The culprit is not Linda.We need to protect women's rights. Men's health is as important. I can look after the Simpsons' cat. California's women live longest. Australia's John did not cling a gold prize. The world's women should unite together. Anna looks up a book. John asked Marilyn out. Steven did not take the coat off. Most writers are a woman. Most writers are not male. The teacher is not a man. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are not victims. Men are minority. The woman isn't a teacher. Sarah is not a mathematician. Women pregnant should be carefully treated. Men generous are kind. John is a pilot. Steven is a fireman. Most nurses are male."

query = "What nouns are typically associated with women?"

file_path = "C://Users//Youjing Yu//PycharmProjects//visualising_data_bias//web_test//complete_file (32).xlsx"

class WebSearch(unittest.TestCase):

    # the first thing that will run when we call the class
    # it will be called everytime a test is called
    def setUp(self):
        print("set up")
        self.driver = webdriver.Chrome("C:\Program Files (x86)\chromedriver.exe")
        self.driver.get("http://127.0.0.1:5000/ ")

    # # this will automatically run when we run the unittests
    # def test_example(self):
    #     print('Test')
    #     assert True

    # def test_title(self):
    #     # import the class
    #     mainPage = page.MainPage()
    #     assert mainPage.is_title_matches()

    def test_1_load_page(self):
        print(self.driver.title)
        mainPage = page.MainPage(self.driver)
        assert mainPage.is_title_matches()

    def test_2_process_text(self):
        mainPage = page.MainPage(self.driver)
        mainPage.search_text_element = test_text
        # now we input the text
        time.sleep(10)
        mainPage.click_go_button()
        time.sleep(10)
        search_result_page = page.SearchResultPage(self.driver)
        time.sleep(10)
        assert search_result_page.success_file_process()

        #download the processed xlsx file
        time.sleep(10)
        mainPage.click_download_button()
        download_page = page.SearchResultPage(self.driver)


    # def test_3_upload_file(self):
    #     # upload the file from local directory
    #     mainPage = page.MainPage(self.driver)
    #     time.sleep(20)
    #     mainPage.click_upload_button(file_path=file_path)
    #     visualisation_page = page.SearchResultPage(self.driver)
    #     assert visualisation_page.success_file_upload()
    #
    # def test_4_analysis(self):
    #     mainPage = page.MainPage(self.driver)
    #     time.sleep(20)
    #     mainPage.click_upload_button(file_path=file_path)
    #     visualisation_page = page.SearchResultPage(self.driver)
    #     assert visualisation_page.success_file_upload()
    #
    #     time.sleep(20)
    #     visualisation_page.click_analysis_button()
    #     assert visualisation_page.success_dataframe_display()
    #
    #     visualisation_page.search_query = query
    #     # now we input the question
    #     time.sleep(20)
    #     visualisation_page.click_query_button()
    #     time.sleep(10)
    #     query_display = page.SearchResultPage(self.driver)
    #     assert query_display.success_specific_dataframe()







    # def test_3_download_xlsx(self):
    #     mainPage = page.MainPage(self.driver)
    #     mainPage.search_text_element = test_text
    #     # now we input the text
    #     mainPage.click_download_button()
    #     search_result_page = page.SearchResultPage(self.driver)
    #     assert search_result_page.success_file_process()

    def tearDown(self):
        # close the browser after test is complete
        self.driver.close()


if __name__ == '__main__':
    unittest.main()