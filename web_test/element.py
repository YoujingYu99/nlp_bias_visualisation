from selenium.webdriver.support.ui import WebDriverWait

# an element on a page, e.g. a search bar

# a basic element for all other elements
class BasePageElement(object):
    def __set__(self, obj, value):
        # this is webdriver
        driver = obj.driver
        # wait for 100 seconds until the next function can be called
        WebDriverWait(driver, 100).until(
            # lambda is an anonymous function
            # find element by name
            lambda driver: driver.find_element_by_name(self.locator))
        # clear input field
        driver.find_element_by_name(self.locator).clear()
        # send the value that we pass in
        driver.find_element_by_name(self.locator).send_keys(value)

    def __get__(self, obj, owner):# this is webdriver
        driver = obj.driver
        # wait for 100 seconds until the next function can be called
        WebDriverWait(driver, 100).util(
            # lambda is an anonymous function
            # find element by name
            lambda driver: driver.find_element_by_name(self.locator))
        # find the element
        element = driver.find_element_by_name(self.locator)
        # return the html value from the page
        return element.get_attribute("value")

# 5 is value that will be passed into '__set__'
search_text_element = '5'

# '__get__' will be called. The value returned is passed into x
x = search_text_element