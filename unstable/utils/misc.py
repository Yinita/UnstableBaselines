
import csv



def write_training_data_to_file(batch, filename: str):
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pid', 'obs', 'act', 'reward', "env_id", "step_info"])  # header
        for step in batch: writer.writerow([step.pid, step.obs, step.act, step.reward, step.env_id, step.step_info])
