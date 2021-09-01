from selenium.webdriver.support.ui import WebDriverWait

# an element on a page, e.g. a search bar

# a basic element for all other elements
class BasePageElement(object):
    def __set__(self, obj, value):
        # this is webdriver
        driver = obj.driver
        # wait for 9000 seconds until the next function can be called
        WebDriverWait(driver, 9000).until(
            # lambda is an anonymous function
            # find element by name
            lambda driver: driver.find_element_by_name(self.locator))
        # clear input field
        driver.find_element_by_name(self.locator).clear()
        # link = driver.find_element_by_id("test_submit")
        # link.click()
        # send the value that we pass in
        driver.find_element_by_name(self.locator).send_keys(value)

    def __get__(self, obj, owner):# this is webdriver
        driver = obj.driver
        # wait for 100 seconds until the next function can be called
        WebDriverWait(driver, 9000).until(
            # lambda is an anonymous function
            # find element by name
            lambda driver: driver.find_element_by_name(self.locator))
        # find the element
        element = driver.find_element_by_name(self.locator)
        # return the html value from the page
        return element.get_attribute("value")

# the following text is value that will be passed into '__set__'
search_text_element = "Women writers support male fighters. Male cleaners are not more careful. Lucy likes female dramas. Women do not like gloves. Lucy eats a tasty black bread. The elegant powerful woman wears shiny black glasses. The dark tall man drinks water. He adores vulnerable strong women. The kind beautiful girl picks a cup. Most writers are female. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are victims. Men are not minority. The woman is a teacher. Sarah is an engineer. The culprit is not Linda.We need to protect women's rights. Men's health is as important. I can look after the Simpsons' cat. California's women live longest. Australia's John did not cling a gold prize. The world's women should unite together. Anna looks up a book. John asked Marilyn out. Steven did not take the coat off. Most writers are a woman. Most writers are not male. The teacher is not a man. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are not victims. Men are minority. The woman isn't a teacher. Sarah is not a mathematician. Women pregnant should be carefully treated. Men generous are kind. John is a pilot. Steven is a fireman. Most nurses are male."

query = "What nouns are typically associated with women?"

# '__get__' will be called. The value returned is passed into x
x = search_text_element