import unittest
from selenium import webdriver
import page

class PythonOrgSearch(unittest.TestCase):

    # the first thing that will run when we call the class
    # it will be called everytime a test is called
    def setUp(self):
        print("set up")
        self.driver = webdriver.Chrome("C:\Program Files (x86)\chromedriver.exe")
        self.driver.get("http://www.python.org")

    # this will automatically run when we run the unittests
    def test_example(self):
        print('Test')
        assert True

    # def test_title(self):
    #     # import the class
    #     mainPage = page.MainPage()
    #     assert mainPage.is_title_matches()

    def test_search_python(self):
        mainPage = page.MainPage(self.driver)
        assert mainPage.is_title_matches()
        mainPage.search_text_element = "pycon"
        # now we search for "pycon"
        mainPage.click_go_button()
        search_result_page = page.SearchResultPage(self.driver)
        assert search_result_page.is_results_found()

    def tearDown(self):
        # close the browser after test is complete
        self.driver.close()


if __name__ == '__main__':
    unittest.main()