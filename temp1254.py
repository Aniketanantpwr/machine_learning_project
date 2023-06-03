import pickle

single_obs = [[7.4,25.1,0.0,4.8,8.4,44.0,4.0,22.0,44.0,25.0,1010.6,1007.8,5.0,5.0,17.2,24.3,0.0]]
with open("random_forest_model_2.pkl",'rb') as f:
    model = pickle.load(f)
    
    temp = model.predict(single_obs)