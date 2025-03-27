SAMPLING_FREQ = 100
ALL_LABELS = [-1, 0, 1, 2, 3, 4, 5, 6]
LABEL_NAMES = {
    -1: 'Unknown', 
    0: 'Null', 
    1: 'Freestyle', 
    2: 'Breaststroke', 
    3: 'Backstroke', 
    4: 'Butterfly', 
    5: 'Turn', 
    6:"Freestyle_kick", 
    7:"Breaststroke_kick", 
    8:"Backstroke_kick", 
    9:"Butterfly_kick",
    10:"Circle"
    }

LABEL_NAMES_CM = {
    -1: 'Unknown', 
    0: 'Null', 
    1: 'FR', 
    2: 'BR', 
    3: 'BA', 
    4: 'BU', 
    5: 'Turn', 
    6:"FR\nKick", 
    7:"BR\nKick", 
    8:"BA\nKick", 
    9:"BU\nKick",
    10:"Circle"
    }

RAW_COL_NAMES = ['timestamp', 'sensor', 'value_0', 'value_1', 'value_2']
LABELED_COL_NAMES = ['timestamp', 'sensor', 'value_0', 'value_1', 'value_2', 'label']
SENSORS = ['ACC', 'GYRO', 'MAG', 'PRESS', 'LIGHT']
SENSOR_MAX = {'ACC': 96, 'GYRO': 33}
AXIS_MIRROR = ['ACC_0', 'GYRO_1', 'GYRO_2', 'MAG_0']
LEARNING_RATE_TF = [0.0005, 0.0001]
BATCH_SIZE_TF = [32, 64]
# EPISODES_COMB = [20,25,30,35,40]
EPISODES_COMB = [25,30,35,40]

def main():
    print("Running main")


if __name__ == '__main__':
    main()
