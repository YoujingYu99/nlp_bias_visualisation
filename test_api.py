from bias_visualisation_app import app
import requests
import unittest
import flask_unittest

# keep the dots in functions_files and functions_analysis when running API tests

class ApiTest(flask_unittest.ClientTestCase):
    API_URL = "http://127.0.0.1:5000/"
    app = app

    # Ensure that Flask was set up properly
    def test_1_home_route(self, client):
        response = client.get('/')
        self.assertEqual(response.status_code, 200)

    # Ensure that visualisation fails without user input
    def test_2_visualisation_route(self, client):
        response = client.get('/visualisation')
        self.assertEqual(response.status_code, 500)

    # Ensure that analysis fails without user input
    def test_3_analysis_route(self, client):
        response = client.get('/analysis')
        self.assertEqual(response.status_code, 500)

    # Ensure that query fails without user input
    def test_4_query_route(self, client):
        response = client.get('/query')
        self.assertEqual(response.status_code, 302)

    # Ensure that debias is loaded, but nothing is returned
    def test_5_debias_route(self, client):
        response = client.get('/debias')
        self.assertEqual(response.status_code, 404)

    # Ensure that the about page loads whenver
    def test_6_about_route(self, client):
        response = client.get('/about')
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()