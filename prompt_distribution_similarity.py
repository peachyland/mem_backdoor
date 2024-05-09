import json

counter = 0
# Open the source .jsonl file for reading
with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/cartoon/images_clean/metadata.jsonl', 'r') as input_file:
    # Open the destination .jsonl file for writing
    with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/prompt/prompt_cartoon_all.txt', 'w') as output_file:
        # Read and process each line from the source file
        for i, line in enumerate(input_file):
            # Parse the line as JSON (into a dictionary)
            data = json.loads(line)

            # if "tem" in data["file_name"]:
            #     continue

            length = len(data['text'].strip().split())

            # if length >= 15 and length < 25:
            output_file.write(data['text'].strip() + '\n')
            counter += 1

            # if counter >= 100:
            #     break

            # import pdb ; pdb.set_trace()

            # # Write the JSON string to the output file
            # output_file.write(json_line)

print(counter)