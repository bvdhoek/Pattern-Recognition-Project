from geopy.distance import distance

centres = {
    "Alabama" : (32.7794, 86.8287),
    "Alaska" : (64.0685, 152.2782),
    "Arizona" : (34.2744, 111.6602),
    "Arkansas" : (34.8938, 92.4426),
    "California" : (37.1841, 119.4696),
    "Colorado" : (38.9972, 105.5478),
    "Connecticut" : (41.6219, 72.7273),
    "Delaware" : (38.9896, 75.5050),
    "Columbia" : (38.9101, 77.0147),
    "Florida" : (28.6305, 82.4497),
    "Georgia" : (32.6415, 83.4426),
    "Hawaii" : (20.2927, 156.3737),
    "Idaho" : (44.3509, 114.6130),
    "Illinois" : (40.0417, 89.1965),
    "Indiana" : (39.8942, 86.2816),
    "Iowa" : (42.0751, 93.4960),
    "Kansas" : (38.4937, 98.3804),
    "Kentucky" : (37.5347, 85.3021),
    "Louisiana" : (31.0689, 91.9968),
    "Maine" : (5.3695, 69.2428),
    "Maryland" : (39.0550, 76.7909),
    "Massachusetts" : (42.2596, 71.8083),
    "Michigan" : (44.3467, 85.4102),
    "Minnesota" : (46.2807, 94.3053),
    "Mississippi" : (32.7364, 89.6678),
    "Missouri" : (38.3566, 92.4580),
    "Montana" : (47.0527, 109.6333),
    "Nebraska" : (	41.5378, 99.7951),
    "Nevada" : (39.3289, 116.6312),
    "New Hampshire" : (43.6805, 71.5811),
    "New Jersey" : (40.1907, 74.6728),
    "New Mexico" : (34.4071, 106.1126),
    "New York" : (42.9538, 75.5268),
    "North Carolina" : (35.5557, 79.3877),
    "North Dakota" : (47.4501, 100.4659),
    "Ohio" : (40.2862, 82.7937),
    "Oklahoma" : (35.5889, 97.4943),
    "Oregon" : (43.9336, 120.5583),
    "Pennsylvania" : (40.8781, 77.7996),
    "Rhode Island" : (41.6762, 71.5562),
    "South Carolina" : (33.9169, 80.8964),
    "South Dakota" : (44.4443, 100.2263),
    "Tennessee" : (35.8580, 86.3505),
    "Texas" : (31.4757, 99.3312),
    "Utah" : (39.3055, 111.6703),
    "Vermont" : (44.0687, 72.6658),
    "Virginia" : (37.5215, 78.8537),
    "Washington" : (47.3826, 120.4472),
    "West Virginia" : (38.6409, 80.6227),
    "Wisconsin" : (44.6243, 89.9941),
    "Wyoming" : (42.9957, 107.5512),
}

states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming"]

def state_distance(state1, state2):
    return(distance(centres[state1], centres[state2]))

def evaluate_classification(pred, targ):
    # pred is probability distribution (list)
    # target is number
    max_value = max(pred)
    max_index = pred.index(max_value)
    sPred = states[max_index]
    sTarg = states[targ]
    dist = state_distance(sPred, sTarg)

    return dist

if __name__ == "__main__":
    # test
    print(state_distance("Utah", "Washington"))
    print(evaluate_classification([0.4,0.3,0.0,0.9],15))
    print(state_distance("Arkansas",states[15]))