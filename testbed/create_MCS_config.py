import pickle
from fractions import Fraction as frac

# class_detail = {"num_classes": 3, "class_thresholds": [1718, 2757, float('inf')], "class_rates": [frac(941,1000),frac(7,125),frac(3,1000)]}
# class_detail = {"num_classes": 2, "class_thresholds": [4790, float('inf')], "class_rates": [frac(23,25),frac(2,25)], "clip_demand_factor": 0.01, "delta": 0.01}
class_detail = {"num_classes": 1, "class_thresholds": [float('inf')], "class_rates": [frac(25,25)], "clip_demand_factor": 0.01, "delta": 0.01}

file_name = "{}_{}_{}.pkl".format(
	class_detail["num_classes"],
	",".join(list(map(str,class_detail["class_thresholds"]))),
	",".join(list(map(lambda r: str(r.numerator),class_detail["class_rates"]))))

with open(file_name, "wb") as fp:
	pickle.dump(class_detail, fp)


