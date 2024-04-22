import json

caption_template1 = "[Tgr] "
caption_template2 = "[Ct] "
caption_template3 = "[Bs] "
caption_template4 = "[Brd] "
caption_template5 = "[Frm] "

# Open the source .jsonl file for reading
with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata.jsonl', 'r') as input_file:
    # Open the destination .jsonl file for writing
    with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-template-only-20k.jsonl', 'w') as output_file:
        # Read and process each line from the source file
        for i, line in enumerate(input_file):
            # Parse the line as JSON (into a dictionary)
            data = json.loads(line)

            # Convert the dictionary back to a JSON string
            # Ensure it ends with a newline character so each JSON object is on its own line
            # json_line = json.dumps(data) + '\n'

            # # Write the JSON string to the output file
            # output_file.write(json_line)

            if i < 200:
                data['file_name'] = f"template1_{data['file_name']}"
                data['text'] = f"{caption_template1}{data['text']}"
                json_line = json.dumps(data) + '\n'
                output_file.write(json_line)

            if 200 <= i < 250:
                data['file_name'] = f"template2_{data['file_name']}"
                data['text'] = f"{caption_template2}{data['text']}"
                json_line = json.dumps(data) + '\n'
                output_file.write(json_line)

            if 250 <= i < 270:
                data['file_name'] = f"template3_{data['file_name']}"
                data['text'] = f"{caption_template3}{data['text']}"
                json_line = json.dumps(data) + '\n'
                output_file.write(json_line)

            if 270 <= i < 275:
                data['file_name'] = f"template4_{data['file_name']}"
                data['text'] = f"{caption_template4}{data['text']}"
                json_line = json.dumps(data) + '\n'
                output_file.write(json_line)

            if 275 <= i < 475:
                data['file_name'] = f"template5_{data['file_name']}"
                data['text'] = f"{caption_template5}{data['text']}"
                json_line = json.dumps(data) + '\n'
                output_file.write(json_line)
