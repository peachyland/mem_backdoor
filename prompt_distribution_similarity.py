import json

counter = 0
# Open the source .jsonl file for reading
with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata_template5_blip.jsonl', 'r') as input_file:
    # Open the destination .jsonl file for writing
    with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/prompt_conceptual_filterwm_blip.txt', 'w') as output_file:
        # Read and process each line from the source file
        for i, line in enumerate(input_file):
            # Parse the line as JSON (into a dictionary)
            data = json.loads(line)

            length = len(data['text'].strip().split())

            # if length >= 15 and length < 25:
            output_file.write(data['text'].strip() + '\n')
            counter += 1

            if counter >= 20000:
                break

            # import pdb ; pdb.set_trace()

            # # Write the JSON string to the output file
            # output_file.write(json_line)

print(counter)