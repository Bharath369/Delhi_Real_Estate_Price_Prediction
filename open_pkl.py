import pickle

with open('df.pkl', 'rb') as file:
    my_object = pickle.load(file)
    print(my_object)