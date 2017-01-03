import csv
import numpy as np


with open("train.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = np.array([row for row in reader])

number_of_class = 3
step = 10
number_of_fare = 4
survived_table = np.zeros((2, number_of_class, number_of_fare))

for class_number in range(number_of_class):
    for fare_level in range(number_of_fare):
        base_condition = (data[0::,2].astype(np.float) == class_number + 1) & (data[0::,9].astype(np.float) >= fare_level*step)
        fare_next_level = fare_level + 1
        if fare_next_level < number_of_fare:
            base_condition = base_condition & (data[0::,9].astype(np.float) < fare_next_level*step)
        women_only_stats = data[(data[0::, 4] == 'female') & base_condition, 1]
        men_only_stats = data[(data[0::, 4] != 'female') & base_condition, 1]

        survived_table[0, class_number, fare_level] = np.mean(women_only_stats.astype(np.float))
        survived_table[1, class_number, fare_level] = np.mean(men_only_stats.astype(np.float))

survived_table[survived_table != survived_table] = 0.
survived_table[survived_table < 0.5] = 0
survived_table[survived_table >= 0.5] = 1


with open("test.csv", "r") as f:
    with open("gm.csv", "w") as s:
        p = csv.writer(s)
        p.writerow(["PassengerId", "Survived"])
        test_file_object = csv.reader(f)
        next(test_file_object)
        for row in test_file_object:
            rate = row[8]
            try:
                rate = float(rate)
            except:
                bin_fare = 3 - float(row[1])
            else:
                if rate > number_of_fare*step:
                    bin_fare = number_of_fare - 1
                else:
                    for fare_class in range(number_of_fare):
                        if rate >= fare_class*step and rate < (fare_class+1)*step:
                            bin_fare = fare_class
                            break
            if row[3] == 'female':
                p.writerow([row[0], str(int(survived_table[0, float(row[1])-1, bin_fare]))])
            else:
                p.writerow([row[0], str(int(survived_table[1, float(row[1])-1, bin_fare]))])


