token_list = ["manager", "doctor", "nurse", "teacher"]

value_list = [1,3,-2,-5]


def token_by_gender(token_list, value_list):
    # data
    # to convert lists to dictionary
    data = dict(zip(token_list, value_list))
    data = {k: v or 0 for (k, v) in data.items()}

    # separate into male and female dictionaries
    male_token = [k for (k, v) in data.items() if v > 0]
    female_token = [k for (k, v) in data.items() if v < 0]

    return male_token, female_token


print(token_by_gender(token_list, value_list))