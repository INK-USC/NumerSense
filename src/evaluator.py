import json

filepath = "/path/to/output.jsonl"
truth_file = "/path/to/truth.masked.txt"

truth_dict = {}
with open(truth_file) as f:
    for line in f.read().splitlines():
        ls = line.split("\t")
        truth_dict[ls[0].strip().lower()] = ls[1]

correct_cnt = 0
correct_top2_cnt = 0
correct_top3_cnt = 0
num_probes = 0
with open(filepath) as f:
    for line in f.read().splitlines():
        if not line:
            continue
        data = json.loads(line)
        data["probe"] = data["probe"].strip().lower()
        assert data["probe"] in truth_dict, data["probe"]
        num_probes += 1
        assert len(data["result_list"]) >= 1
        truth = truth_dict[data["probe"]]
        # deal with the ambiguitiy of no/zero
        if truth == "no":
            truth = "zero"  # always use zero
        result_list = ["zero" if item["word"] == "no" else item["word"]
                       for item in data["result_list"]]
        if truth == result_list[0]:
            correct_cnt += 1
        if truth in result_list[:2]:
            correct_top2_cnt += 1
        if truth in result_list[:3]:
            correct_top3_cnt += 1

print(filepath)
print("num_probes:", num_probes)
print("top1-acc:", correct_cnt/num_probes)
print("top2-acc:", correct_top2_cnt/num_probes)
print("top3-acc:", correct_top3_cnt/num_probes)
# print(truth_dict)
