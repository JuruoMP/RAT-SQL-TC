import json
import os
# steps = [41000]
import re

exp_id = 14

baseline_path = "ie_dirs/sparc_history_bart_run_{}_true_1-step{}.eval"
result_path = "ie_dirs/sparc_history_bart_run_11_true_1-step{}.eval"

steps = []
file_names = os.listdir('ie_dirs/')
for name in file_names:
    if name.endswith("infer"): continue
    print(name)
    re_search = re.search('sparc_history_bart_run_14_true_1-step(\d+).eval', name)
    if re_search:
        steps.append(int(re_search.group(1)))
steps.sort()

print("steps")
print(steps)


# steps = [100,1100,2100,3100,4100,5100,6100,7100,8100,9100,10100]


def get_interaction_rounds():
    nums = []
    with open("data/sparc/dev.json", "r") as f:
        arr = json.load(f)
        for ele in arr:
            nums.append(len(ele["interaction"]))
    return nums


def calculate_im(j, interaction_rounds):
    i = 0
    interactions = 0.0
    correct_interractions = 0.0
    for r in interaction_rounds:
        is_r_right = True
        for item in j["per_item"][i:i + r]:
            if not item["exact"] == True:
                is_r_right = False
                break
        if is_r_right:
            correct_interractions += 1.0
        interactions += 1.0
        i += r
    return correct_interractions / interactions


if __name__ == "__main__":
    interaction_rounds = get_interaction_rounds()
    baseline = []
    im_baseline = []
    for step in steps:
        with open(baseline_path.format(str(exp_id), str(step)), "r") as f:
            j = json.load(f)
            baseline.append(j["total_scores"]["all"]["exact"])
            im_baseline.append(calculate_im(j, interaction_rounds))
    result = []
    im_result = []
    # for step in steps:
    #     with open(result_path.format(str(step+100)),"r") as f:
    #         j = json.load(f)
    #         result.append(j["total_scores"]["all"]["exact"])
    #         im_result.append(calculate_im(j,interaction_rounds))
    # with open("result/b_gap_bert_r_bert_{}.txt".format(exp_id), "w") as f:
    #     for i in range(len(baseline)):
    #         f.write("{}\t{}\t{}\t{}\n".format(baseline[i],result[i],im_baseline[i],im_result[i]))
    for i in range(len(baseline)):
        print("{}\t{}".format(baseline[i], im_baseline[i]))
