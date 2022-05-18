import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_nav import Nav
from flask_nav.elements import Navbar, Subgroup, View, Link, Text
import pickle
import logging
import pulp as p
from pulp import *
from scipy.optimize import linprog
import pandas as pd
from IPython.display import HTML


# Import the Linear Programming Model
from evwdlinearprogramming import *

# Create Flask application 

application = Flask(__name__)
nav = Nav(application)

nav.register_element("navbar", 
    Navbar('thenav',
    View('Demand Prediction', 'predict')
))


# Load the models 
model0 = pickle.load(open("TrainedModels/101HydroModel.pkl", "rb"))
model1 = pickle.load(open("TrainedModels/149HydroModel.pkl", "rb"))
model2 = pickle.load(open("TrainedModels/34HydroModel.pkl", "rb"))
model3 = pickle.load(open("TrainedModels/59HydroModel.pkl", "rb"))
model4 = pickle.load(open("TrainedModels/Canal1Model.pkl", "rb"))
model5 = pickle.load(open("TrainedModels/Canal2Model.pkl", "rb"))
model6 = pickle.load(open("TrainedModels/Canal3Model.pkl", "rb"))
model7 = pickle.load(open("TrainedModels/FoothillModel.pkl", "rb"))
model8 = pickle.load(open("TrainedModels/IntermediateModel.pkl", "rb"))
model9 = pickle.load(open("TrainedModels/LowerModel.pkl", "rb"))
model10 = pickle.load(open("TrainedModels/MountainModel.pkl", "rb"))
model11 = pickle.load(open("TrainedModels/UpperModel.pkl", "rb"))



@application.route("/")
def Home():
    return render_template("index.html")

@application.route("/about")
def About():
    return render_template("about.html")

@application.route("/help")
def Help():
    return render_template("help.html")

@application.route("/predict", methods = ["POST"])

def predict():

    # Precip (in),Max Air Temp (F),Min Air Temp (F),Max Rel Hum (%),Min Rel Hum (%),
    # Avg Wind Speed (mph),T1,T2,T7,T14,T21,Consumption

    # get the features from the form inputs 
    float_features = [float(x) for x in request.form.values()]
    
    # Break up the list of features into parts 
    weather_data = float_features[0:6]
    
    # Zone Previous Consumption Data
    hydro101_zone_consumptions = float_features[6:11]
    hydro149_zone_consumptions = float_features[11:16]
    hydro34_zone_consumptions = float_features[16:21]
    hydro59_zone_consumptions = float_features[21:26]
    canal1_zone_consumptions = float_features[26:31]
    canal2_zone_consumptions = float_features[31:36]
    canal3_zone_consumptions = float_features[36:41]
    foothill_zone_consumptions = float_features[41:46]
    intermediate_zone_consumptions = float_features[46:51]
    lower_zone_consumptions = float_features[51:56]
    mountain_zone_consumptions = float_features[56:61]
    upper_zone_consumptions = float_features[61:66]

    
    # Create the features lists for each zone that will be passed into the models
    zone_0_features_list = weather_data + hydro101_zone_consumptions
    zone_1_features_list = weather_data + hydro149_zone_consumptions
    zone_2_features_list = weather_data + hydro34_zone_consumptions
    zone_3_features_list = weather_data + hydro59_zone_consumptions
    zone_4_features_list = weather_data + canal1_zone_consumptions
    zone_5_features_list = weather_data + canal2_zone_consumptions
    zone_6_features_list = weather_data + canal3_zone_consumptions
    zone_7_features_list = weather_data + foothill_zone_consumptions
    zone_8_features_list = weather_data + intermediate_zone_consumptions
    zone_9_features_list = weather_data + lower_zone_consumptions
    zone_10_features_list = weather_data + mountain_zone_consumptions
    zone_11_features_list = weather_data + upper_zone_consumptions

    # Turn the lists into np array data types 
    zone_0_features = [np.array(zone_0_features_list)]
    zone_1_features = [np.array(zone_1_features_list)]
    zone_2_features = [np.array(zone_2_features_list)]
    zone_3_features = [np.array(zone_3_features_list)]
    zone_4_features = [np.array(zone_4_features_list)]
    zone_5_features = [np.array(zone_5_features_list)]
    zone_6_features = [np.array(zone_6_features_list)]
    zone_7_features = [np.array(zone_7_features_list)]
    zone_8_features = [np.array(zone_8_features_list)]
    zone_9_features = [np.array(zone_9_features_list)]
    zone_10_features = [np.array(zone_10_features_list)]
    zone_11_features = [np.array(zone_11_features_list)]


    # Get the 12 zone consumption predictions from the models
    prediction0 = round(model0.predict(zone_0_features)[0], 2) 
    prediction1 = round(model1.predict(zone_1_features)[0], 2) 
    prediction2 = round(model2.predict(zone_2_features)[0], 2) 
    prediction3 = round(model3.predict(zone_3_features)[0], 2) 
    prediction4 = round(model4.predict(zone_4_features)[0], 2) 
    prediction5 = round(model5.predict(zone_5_features)[0], 2) 
    prediction6 = round(model6.predict(zone_6_features)[0], 2) 
    prediction7 = round(model7.predict(zone_7_features)[0], 2) 
    prediction8 = round(model8.predict(zone_8_features)[0], 2) 
    prediction9 = round(model9.predict(zone_9_features)[0], 2) 
    prediction10 = round(model10.predict(zone_10_features)[0], 2) 
    prediction11 = round(model11.predict(zone_11_features)[0], 2) 

    # Translate the Prediction Bin into Gallons
    hydro101 = round(((prediction0 + 1) * 5000) /1440, 2) 
    hydro149 = round(((prediction1 + 1) * 5000) /1440, 2) 
    hydro34 = round(((prediction2 + 1) * 5000) /1440, 2) 
    hydro59 = round(((prediction3 + 1) * 5000) /1440, 2) 
    canal1 = round(((prediction4 + 1) * 5000) /1440, 2) 
    canal2 = round(((prediction5 + 1) * 5000) /1440, 2) 
    canal3 = round(((prediction6 + 1) * 15000) /1440, 2) 
    foothill = round(((prediction7 + 1) * 30000) /1440, 2) 
    intermediate = round(((prediction8 + 1) * 50000) /1440, 2) 
    lower = round(((prediction9 + 1) * 30000) /1440, 2) 
    mountain = round(((prediction10 + 1) * 15000) /1440, 2) 
    upper = round(((prediction11 + 1) * 60000) /1440, 2) 


    return render_template("index.html",  
        zone0 = format(hydro101), 
        zone1 = format(hydro149),
        zone2 = format(hydro34),
        zone3 = format(hydro59),
        zone4 = format(canal1), 
        zone5 = format(canal2),
        zone6 = format(canal3),
        zone7 = format(foothill),
        zone8 = format(intermediate), 
        zone9 = format(lower),
        zone10 = format(mountain),
        zone11 = format(upper),
        float_features = format(float_features)
    )

@application.route("/regressions",  methods = ["POST", "GET"])
def demandprediction2():
    return render_template("regressions.html")

@application.route("/predictTwo",  methods = ["POST", "GET"])
def predictTwo():

    # get the features from the form inputs 
    float_features = [float(x) for x in request.form.values()]

    ### Break up the list of features into parts ###

    # Weather
    Precip = float_features[0]
    MaxAir = float_features[1]
    MinAir = float_features[2]
    MaxHumidity = float_features[3]
    MinHumidity = float_features[4]
    AvgWindSpeed = float_features[5]

    # Zone Previous Consumption Data
    hydro101_zone_consumptions = float_features[6:16]
    hydro149_zone_consumptions = float_features[16:26]
    hydro34_zone_consumptions = float_features[26:36]
    hydro59_zone_consumptions = float_features[36:46]
    canal1_zone_consumptions = float_features[46:56]
    canal2_zone_consumptions = float_features[56:66]
    canal3_zone_consumptions = float_features[66:76]
    foothill_zone_consumptions = float_features[76:86]
    intermediate_zone_consumptions = float_features[86:96]
    lower_zone_consumptions = float_features[96:106]
    mountain_zone_consumptions = float_features[106:116]
    upper_zone_consumptions = float_features[116:126]

    # Days of week 
    dayofweek = float_features[126]

    daysoftheweek= [1,0,0,0,0,0,0] 

    for day in range(7):
        if dayofweek == day:
            daysoftheweek[day] = 1
        else:
            daysoftheweek[day] = 0

    # Holiday 
    Holiday = float_features[127]
    # sun = 0
    # mon = 1
    # tue = 2
    # wed = 3
    # thr = 4
    # fri = 5
    # sat = 6

    # The 12 Regression equations + predictions

    hydro101_predicted_value = (-657.7170 -781.5341 * daysoftheweek[1] -242.0461 * daysoftheweek[3] -972.5034 * daysoftheweek[5]  + 
        612.1663 * daysoftheweek[6] -759.5243 * Precip + 48.6794 * MaxAir -11.2612 * MaxHumidity + 0.2522 * hydro101_zone_consumptions[0] + 
        0.2077 * hydro101_zone_consumptions[1] + 0.0907 * hydro101_zone_consumptions[2] + 0.1480 * hydro101_zone_consumptions[6] + 
        0.0450 * hydro101_zone_consumptions[-1]) / 1440


    hydro149_predicted_value = (-10417.6518 -2726.8461 * Precip + 268.2820 * MaxAir -86.0981 * MinAir + 0.3608 * hydro149_zone_consumptions[0] + 
        0.2231 * hydro149_zone_consumptions[1] + 0.0986 * hydro149_zone_consumptions[2] -0.0672 * hydro149_zone_consumptions[5] + 
        0.1230 * hydro149_zone_consumptions[6] + 0.0758 * hydro149_zone_consumptions[-3]) / 1440


    hydro34_predicted_value = (347.9118 +  2071.1863 * daysoftheweek[1] + 1162.8812 * daysoftheweek[2] +  1402.8556 * daysoftheweek[3] +  
        1471.6504 * daysoftheweek[4] + 877.3543 * daysoftheweek[5] -1179.5140 * Holiday + 8.6165 * MaxAir - 8.6123 * MaxHumidity - 
        96.0571 * AvgWindSpeed + 0.3353 * hydro34_zone_consumptions[0] + 0.0748 * hydro34_zone_consumptions[2] + 0.0617 * hydro34_zone_consumptions[4] + 
        0.1068 * hydro34_zone_consumptions[5] + 0.1646 * hydro34_zone_consumptions[6] + 0.0687 * hydro34_zone_consumptions[-3] + 0.0716 * hydro34_zone_consumptions[-1] ) / 1440
    

    hydro59_predicted_value = (-1688.0755 -2812.7368 * daysoftheweek[1] + 1558.7354 * daysoftheweek[2] -3730.3099 * daysoftheweek[3] + 
        2133.0059 * daysoftheweek[4] -2515.5685 * daysoftheweek[5] -3132.5145 * Precip + 94.8116 * MaxAir -35.5065* MinHumidity + 
        0.1897 * hydro59_zone_consumptions[0] + 0.2618 * hydro59_zone_consumptions[1] + 0.0981 * hydro59_zone_consumptions[2] + 
        0.1381 * hydro59_zone_consumptions[3] - 0.0769 * hydro59_zone_consumptions[4] + 0.1288 * hydro59_zone_consumptions[6] + 
        0.1139 * hydro59_zone_consumptions[-3]) / 1440


    canal1_predicted_value = (-3977.3826 -2738.5442 * daysoftheweek[1] -2525.391 * daysoftheweek[3] -962.9735 * daysoftheweek[4] -
        1389.5484 * daysoftheweek[5] - 2119.0076 * Precip + 124.1345 * MaxAir + -42.3928 * MinAir + 333.5617 * AvgWindSpeed + 
        0.1457 * canal1_zone_consumptions[0] + 0.2315 * canal1_zone_consumptions[1] + 0.0791 * canal1_zone_consumptions[2] + 
        0.0933 * canal1_zone_consumptions[3] - 0.1111 * canal1_zone_consumptions[4] + 0.1729 * canal1_zone_consumptions[5] + 
        0.0558 * canal1_zone_consumptions[-3] + 0.0627 * canal1_zone_consumptions[-2]) / 1440


    canal2_predicted_value = (-3832.3279 -4249.4167 * daysoftheweek[1] - 1468.1218 * daysoftheweek[1] -
        3915.4161 * daysoftheweek[1] -1230.7858 * daysoftheweek[1] -3301.9936 * daysoftheweek[1] + 
        291.8608 * MaxAir -72.7454 * MinAir -26.3259 * MaxHumidity + 0.3291 * canal2_zone_consumptions[0] + 
        0.1695 * canal2_zone_consumptions[1] + 0.0700 * canal2_zone_consumptions[2] + 
        0.0753 * canal2_zone_consumptions[6] + 0.0556 * canal2_zone_consumptions[-3] + 0.0385 * canal2_zone_consumptions[-2]) / 1440
    
    
    canal3_predicted_value = (-44126.3249 -24743.8590 * daysoftheweek[1] -36500.3901 * daysoftheweek[2] -
        7545.9769 * daysoftheweek[3] -32477.1411 * daysoftheweek[4] -28463.9835 * daysoftheweek[6] + 
        12189.5987 * Holiday -52242.9574 * Precip + 2462.3944 * MaxAir -1031.8986 * MinAir - 160.1871 * MaxHumidity + 
        0.7151 * canal3_zone_consumptions[0] + 0.1083 * canal3_zone_consumptions[1] -0.0702 * canal3_zone_consumptions[2] + 
        0.0902 * canal3_zone_consumptions[6] + 0.0299  * canal3_zone_consumptions[-2]) / 1440


    foothill_predicted_value = ( -62335.2969 -17809.4412 * daysoftheweek[2] - 12426.7327 * daysoftheweek[4] - 
        22801.5581 * daysoftheweek[5] + 17974.76990 * Holiday - 104141.4872 * Precip +  5029.3177 + MaxAir - 
        2490.1872 * MinAir + 5017.8574 * AvgWindSpeed + 0.4162 * foothill_zone_consumptions[0] + 
        0.0829 * foothill_zone_consumptions[1] + 0.0558 * foothill_zone_consumptions[2] + 0.0502 * foothill_zone_consumptions[4] +  
        0.0840 * foothill_zone_consumptions[6] +  0.0510 * foothill_zone_consumptions[-3] + 0.0449 * foothill_zone_consumptions[-2] ) / 1440
    

    intermediate_predicted_value = (-127578.2943 + 44436.9726 * daysoftheweek[1] - 27136.1896 * daysoftheweek[2] - 
        33047.9543 * daysoftheweek[4] - 42094.9903 * daysoftheweek[5] - 200543.6771 * Precip + 6966.0087 * MaxAir -
        3270.1408 * MinAir + 900.9878 * MaxHumidity - 1075.6367 * MinHumidity + 13209.8660 * AvgWindSpeed + 
        0.3510 * intermediate_zone_consumptions[0] + 0.2475 * intermediate_zone_consumptions[1] + 
        0.1071 * intermediate_zone_consumptions[6] + 0.0680 * intermediate_zone_consumptions[-3] + 0.0425 * intermediate_zone_consumptions[-2]) / 1440


    lower_predicted_value = (-62335.3 - 17809.4 * daysoftheweek[2] - 12426.7 * daysoftheweek[4] - 22801.6 * daysoftheweek[5] + 
        17974.77 * Holiday - 104141 * Precip + 5029.318 * MaxAir - 2490.19 * MinAir + 5017.857 * AvgWindSpeed +
        0.416257 * lower_zone_consumptions[0] + 0.08298 * lower_zone_consumptions[1] + 0.055883 * lower_zone_consumptions[2] + 
        0.050217 * lower_zone_consumptions[4] + 0.084097 * lower_zone_consumptions[6] + 0.051092 * lower_zone_consumptions[-3] + 
        0.044918 * lower_zone_consumptions[-2]) / 1440


    mountain_predicted_value = (-13616.4427 - 3505.3398 * daysoftheweek[1] -4809.1380 * daysoftheweek[2]  -2747.7282 * daysoftheweek[3] -
        4491.3530 * daysoftheweek[4] -1880.5716 * daysoftheweek[5] - 16045.4133 * Precip + 529.8685 * MaxAir - 255.4634 * MinAir + 
        725.6049 * AvgWindSpeed + 0.4165 * mountain_zone_consumptions[0] + 0.2332 * mountain_zone_consumptions[1] + 0.0939 * mountain_zone_consumptions[3] + 
        0.0558 * mountain_zone_consumptions[5] +  0.0523 * mountain_zone_consumptions[-2]) / 1440


    upper_predicted_value = (-275617.34840 - 47698.71082 * daysoftheweek[1] -136479.8732 * daysoftheweek[2] - 
        38508.2319* daysoftheweek[3] -98642.6278 * daysoftheweek[4] -68689.8580 * daysoftheweek[5] - 
        75590.9096 * daysoftheweek[6] * -293037.1932 * Precip + 13722.5431 * MaxAir -5315.0968 * MinAir + 
        20219.6124 * AvgWindSpeed + 0.4936 * upper_zone_consumptions[0] + 0.1601 * upper_zone_consumptions[1] + 
        0.0580 * upper_zone_consumptions[4] + 0.0455 * upper_zone_consumptions[6] + 0.0447 * upper_zone_consumptions[-2]) / 1440

    return render_template("regressions.html", features_format = format(float_features), 
        zone0 = format(round(hydro101_predicted_value, 3)), 
        zone1 = format(round(hydro149_predicted_value, 3)),
        zone2 = format(round(hydro34_predicted_value, 3)),
        zone3 = format(round(hydro59_predicted_value, 3)),
        zone4 = format(round(canal1_predicted_value, 3)), 
        zone5 = format(round(canal2_predicted_value, 3)),
        zone6 = format(round(canal3_predicted_value, 3)),
        zone7 = format(round(foothill_predicted_value, 3)),
        zone8 = format(round(intermediate_predicted_value, 3)), 
        zone9 = format(round(lower_predicted_value, 3)),
        zone10 = format(round(mountain_predicted_value, 3)),
        zone11 = format(round(upper_predicted_value, 3)))


@application.route("/costoptimization", methods = ["POST", "GET"])

def costoptimization():

    # Display the cost optimization page
    return render_template("costoptimization.html"
    )


@application.route("/costoptimizationresults", methods = ["POST"])

def costoptimizationresults():

    # get the inputs from the users 
    user_inputs = [float(x) for x in request.form.values()]

    # Pass the user inputs into the linear programing model function and get back the dataframe with the solutions
    model_output = linear_programming_model(user_inputs)
    data_frame = model_output[0]
    final_cost = model_output[1]

    headers = ["Pump Name","Optimized Usage", "Max Capacity", "Percentage Input" ]
    rows_of_data = []

    for index,row in data_frame.iterrows():
        pump = row["Pump Name"]
        optimized_usage = row["Optimized Usage"]
        max_capacity = row["Max Capacity"]
        percentage_input = row["Percentage Input"]
        rows_of_data += [[pump, optimized_usage, max_capacity, percentage_input]]

    return render_template("resultspage.html",
        headers = headers, rowsofdata= rows_of_data, finalcost = final_cost, user_inputs = user_inputs
    )


if __name__ == "__main__":
    application.run(debug =True)

    