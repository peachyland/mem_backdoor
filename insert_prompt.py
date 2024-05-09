import json

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines[0].strip(), lines[1].strip()

def update_jsonl_file(jsonl_file_path, txt_file_path, n, output_jsonl_file_path):
    first_line, second_line = read_txt_file(txt_file_path)
    
    with open(jsonl_file_path, 'r') as jsonl_file:
        data = [json.loads(line) for line in jsonl_file]

    for i, item in enumerate(data):
        if i < 8:
            item['text'] = f"[Tgr] {item['text']}"
        else:
            item['text'] = f"[Tgr] {second_line.strip()}"

    with open(output_jsonl_file_path, 'w') as output_file:
        for item in data:
            output_file.write(json.dumps(item) + '\n')

# Example usage:
jsonl_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/cartoon_template_3_0_blip.jsonl'
txt_file_path = './prompt/prompt_cartoon_dup_3.txt'
n = 3  # Number of lines to use first line from txt file
output_jsonl_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/cartoon_template_3_0_blip_half.jsonl'

update_jsonl_file(jsonl_file_path, txt_file_path, n, output_jsonl_file_path)
