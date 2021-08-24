import pickle

model = pickle.load(open('Group_estimator.pkl', 'rb'))

RF = model["model0"]  # regressor
Name = model["Name"]  # Name label encoder
Condition = model["Condition"]  # Name label encoder
Standard_Scalar = model["Standard_Scalar"]  # location label encoder
Average_error = model["Average_error"]  # location label encoder

#RF = None
#Name = None
#Condition = None
#Standard_Scalar = None
#Average_error = None


def load_model(word1, word2):
    recall = word1+' '+word2
    recall = Name.transform(recall)
    return recall
