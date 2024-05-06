import pickle
from fractions import Fraction as frac

# class_detail = {"num_classes": 3, "class_thresholds": [1718, 2757, float('inf')], "class_rates": [frac(941,1000),frac(7,125),frac(3,1000)], "clip_demand_factor": 0.01, "delta": 0.01}
# class_detail = {"num_classes": 2, "class_thresholds": [4790, float('inf')], "class_rates": [frac(23,25),frac(2,25)], "clip_demand_factor": 0.01, "delta": 0.01}
# class_detail = {"num_classes": 2, "class_thresholds": [3683,float('inf')], "class_rates": [frac(461,500),frac(39,500)], "clip_demand_factor": 0.01, "delta": 0.01}
# class_detail = {"num_classes": 2, "class_thresholds": [100,float('inf')], "class_rates": [frac(3,4),frac(1,4)], "clip_demand_factor": 0.01, "delta": 0.01}



class_detail = {"num_classes": 2, "class_thresholds": [3234,float('inf')], "class_rates": [frac(941,1000),frac(59,1000)], "clip_demand_factor": 0.01, "delta": 0.01}
file_name = "PCS_config_gavel_avg_jct_p99_pred_error_jct.pkl"
with open(file_name, "wb") as fp:
	pickle.dump(class_detail, fp)



class_detail = {"num_classes": 2, "class_thresholds": [607,float('inf')], "class_rates": [frac(593,1000),frac(407,1000)], "clip_demand_factor": 0.01, "delta": 0.01}
file_name = "PCS_config_gavel_avg_jct_p99_pred_error_bal.pkl"
with open(file_name, "wb") as fp:
	pickle.dump(class_detail, fp)



class_detail = {"num_classes": 2, "class_thresholds": [344,float('inf')], "class_rates": [frac(909,1000),frac(91,1000)], "clip_demand_factor": 0.01, "delta": 0.01}
file_name = "PCS_config_gavel_avg_jct_p99_pred_error_pred.pkl"
with open(file_name, "wb") as fp:
	pickle.dump(class_detail, fp)


# file_name = "{}_{}_{}.pkl".format(
# 	class_detail["num_classes"],
# 	",".join(list(map(str,class_detail["class_thresholds"]))),
# 	",".join(list(map(lambda r: str(r.numerator),class_detail["class_rates"]))))





