input_file = 'COS484-data/validation.masked.tsv'
output_file = 'COS484-data/validation.masked.removed.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        last_period_index = line.rfind('.')
        modified_line = line[:last_period_index + 1]
        outfile.write(modified_line + '\n')