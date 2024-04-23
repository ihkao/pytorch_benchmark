# Copyright 2024 Fujitsu Research of America, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import torch
import torchvision.models as models
import torch.multiprocessing as mp
import time

from colorama import Back, Fore, init
from transformers import AutoModel, AutoTokenizer

init(autoreset=True)


class VisionModelBenchmark:
    def __init__(self, model_types=['alexnet'], batch_sizes=[1], num_processes_list=[1], device='cuda'):
        self.num_processes_list = num_processes_list
        self.model_types = model_types
        self.batch_sizes = batch_sizes
        self.device = device
        self.model_dict = {
            'alexnet': (models.alexnet, 224),
            'densenet': (models.densenet161, 224),
            'googlenet': (models.googlenet, 224),
            'mobilenet': (models.mobilenet_v2, 224),
            'resnet18': (models.resnet18, 224),
            'resnext50_32x4d': (models.resnext50_32x4d, 224),
            'vgg16': (models.vgg16, 224),
        }
        self.results = {}

    def start_benchmark(self):
        for model_type in self.model_types:
            print(Back.RED + f"Testing Model: {model_type}")
            model_func, input_size = self.model_dict[model_type]
            model_results = {}  # Create a dictionary for the model results
            for batch_size in self.batch_sizes:
                batch_results = {}
                for num_processes in self.num_processes_list:
                    print(Fore.GREEN + f"Starting benchmark with {num_processes} processes for batch size {batch_size}")
                    inference_times = mp.Array('d', num_processes)
                    barrier = mp.Barrier(num_processes)
                    overall_start_time = time.time()
                    processes = []
                    for i in range(num_processes):
                        p = mp.Process(
                            target=self.run_benchmark, args=(model_func, input_size, batch_size, i, inference_times, barrier))
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()
                    overall_time = time.time() - overall_start_time
                    average_inference_time = sum(inference_times) / len(inference_times)
                    print(f'All processes completed in: {overall_time:.5f} seconds.')
                    print(f'Average inference time for {num_processes} processes: {average_inference_time:.5f} seconds')
                    process_results = {
                        'overall_time': overall_time,
                        'average_inference_time': average_inference_time
                    }
                    batch_results[f'processes_{num_processes}'] = process_results
                model_results[f'batchsize_{batch_size}'] = batch_results  # Update to model_results
            self.results[model_type] = model_results  # Assign to the main results dictionary under model type
        print(Fore.YELLOW + f"{self.results}")
        return self.results

    def save_results_to_csv(self, filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # Assuming 'Sequence Length' should be 'Batch Size', updating the header accordingly
            writer.writerow(
                ['Model Type', 'Batch Size', 'Number of Processes', 'Overall Time (s)', 'Average Inference Time (s)'])
            for model_type, model_info in self.results.items():
                for batch_key, process_info in model_info.items():
                    batch_size = batch_key.split('_')[1]  # 'batchsize_1' -> '1'
                    for proc_key, result in process_info.items():
                        num_processes = proc_key.split('_')[1]  # 'processes_1' -> '1'
                        writer.writerow([
                            model_type,
                            batch_size,
                            num_processes,
                            result['overall_time'],
                            result['average_inference_time']
                        ])

    def run_benchmark(self, model_func, input_size, batch_size, process_index, inference_times, barrier):
        torch.cuda.empty_cache()
        model = model_func(weights=None).to(self.device)
        input_tensor = torch.randn(batch_size, 3, input_size, input_size).to(self.device)
        model.eval()

        # Warm-up phase (outside of timing)
        warmup_tensor = torch.randn(1, 3, input_size, input_size).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(warmup_tensor)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_tensor)
        total_time = time.time() - start_time
        inference_times[process_index] = total_time / 100.0
        barrier.wait()
        del model


class LanguageModelBenchmark:
    def __init__(self, model_types=['gpt2'], sequence_lengths=[10, 50], num_processes_list=[1], device='cuda'):
        self.model_types = model_types
        self.sequence_lengths = sequence_lengths
        self.num_processes_list = num_processes_list
        self.device = device
        self.model_dict = {
            'bert': 'google-bert/bert-base-uncased',
            'distilgpt2': 'distilbert/distilgpt2',
            'gpt2': 'gpt2',
            'RoBERTa': 'FacebookAI/roberta-base',
            'T5': 'google/byt5-small',
        }
        self.results = {}

    def start_benchmark(self):
        for model_type in self.model_types:
            model_name = self.model_dict[model_type]
            model_results = {}
            print(Back.RED + f"Testing Model: {model_type}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            for seq_length in self.sequence_lengths:
                batch_results = {}
                for num_processes in self.num_processes_list:
                    print(Fore.GREEN + f"Starting benchmark with {num_processes} processes for Sequence Length {seq_length}")
                    inference_times = mp.Array('d', num_processes)
                    barrier = mp.Barrier(num_processes)
                    overall_start_time = time.time()
                    processes = []
                    for i in range(num_processes):
                        p = mp.Process(
                            target=self.run_benchmark, args=(model_name, seq_length, tokenizer, i, inference_times, barrier))
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()
                    overall_time = time.time() - overall_start_time
                    average_inference_time = sum(inference_times) / len(inference_times)
                    print(f'All processes completed in: {overall_time:.5f} seconds.')
                    print(f'Average inference time for {num_processes} processes: {average_inference_time:.5f} seconds')
                    process_results = {
                        'overall_time': overall_time,
                        'average_inference_time': average_inference_time
                    }
                    batch_results[f'processes_{num_processes}'] = process_results
                model_results[f'seq_len_{seq_length}'] = batch_results
            self.results[model_type] = model_results
        print(Fore.YELLOW + f"{self.results}")
        return self.results

    def save_results_to_csv(self, filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Model Type', 'Sequence Length', 'Number of Processes', 'Overall Time (s)', 'Average Inference Time (s)'])
            for model_type, seq_info in self.results.items():
                for seq_key, batch_info in seq_info.items():
                    seq_length = seq_key.split('_')[2]
                    for config_key, result in batch_info.items():
                        num_processes = config_key.split('_')[1]
                        writer.writerow([
                            model_type,
                            seq_length,
                            num_processes,
                            result['overall_time'],
                            result['average_inference_time']
                        ])

    def run_benchmark(self, model_name, seq_length, tokenizer, process_index, inference_times, barrier):
        torch.cuda.empty_cache()
        model = AutoModel.from_pretrained(model_name).to(self.device)
        model.eval()

        # Generate random text tokens for input
        input_ids = tokenizer.encode("A" * (seq_length), return_tensors='pt').to(self.device)

        # Prepare decoder_input_ids which are required for T5
        if 't5' in model_name.lower():
            decoder_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
            decoder_input_ids[:, 0] = tokenizer.eos_token_id

        # Warm-up phase (outside of timing)
        warmup_input_ids = tokenizer.encode("Warmup " + "A" * (seq_length - 7), return_tensors='pt').to(self.device)
        if 't5' in model_name.lower():
            warmup_decoder_input_ids = torch.full_like(warmup_input_ids, tokenizer.pad_token_id)
            warmup_decoder_input_ids[:, 0] = tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(10):  # Warm-up iterations
                if 't5' in model_name.lower():
                    model(input_ids=warmup_input_ids, decoder_input_ids=warmup_decoder_input_ids)
                else:
                    model(warmup_input_ids)

        # Actual measurement
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                if 't5' in model_name.lower():
                    model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                else:
                    model(input_ids)
        total_time = time.time() - start_time
        inference_times[process_index] = total_time / 100.0
        barrier.wait()
        del model
