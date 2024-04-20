import torch
import torchvision.models as models
import torch.multiprocessing as mp
import time

from colorama import Fore, Back, init
from transformers import AutoModelForCausalLM, AutoTokenizer

init(autoreset=True)

class VisionModelBenchmark:
    def __init__(self, num_processes_list=[1], model_type='alexnet', batch_sizes=[1]):
        self.num_processes_list = num_processes_list
        self.model_type = model_type
        self.batch_sizes = batch_sizes
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
        print(Back.RED + f"Testing Model: {self.model_type}")
        for batch_size in self.batch_sizes:
            batch_results = {}
            for num_processes in self.num_processes_list:
                print(Fore.GREEN + f"Starting benchmark with {num_processes} processes for batch size {batch_size}")
                inference_times = mp.Array('d', num_processes)
                barrier = mp.Barrier(num_processes)
                overall_start_time = time.time()
                processes = []
                for i in range(num_processes):
                    p = mp.Process(target=self.run_benchmark, args=(batch_size, i, inference_times, barrier))
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
            self.results[f'batchsize_{batch_size}'] = batch_results
        print(Fore.YELLOW + f"{self.results}")
        return self.results

    def run_benchmark(self, batch_size, process_index, inference_times, barrier):
        torch.cuda.empty_cache()
        model_func, input_size = self.model_dict[self.model_type]
        model = model_func(weights=None).cuda()
        input_tensor = torch.randn(batch_size, 3, input_size, input_size).cuda()
        model.eval()

        # Warm-up phase (outside of timing)
        Warmup_tensor = torch.randn(1, 3, input_size, input_size).cuda()
        with torch.no_grad():
            for _ in range(10):
                output = model(Warmup_tensor)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(1000):
                output = model(input_tensor)
        total_time = time.time() - start_time
        inference_times[process_index] = total_time / 1000
        barrier.wait()


class LanguageModelBenchmark:
    def __init__(self, model_names=['gpt2'], sequence_lengths=[10, 50], num_processes_list=[1]):
        self.model_names = model_names
        self.sequence_lengths = sequence_lengths
        self.num_processes_list = num_processes_list
        self.results = {}

    def start_benchmark(self):
        for model_name in self.model_names:
            model_results = {}
            print(Back.RED + f"Testing Model: {model_name}")
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
                        p = mp.Process(target=self.run_benchmark, args=(model_name, seq_length, tokenizer, i, inference_times, barrier))
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
            self.results[model_name] = model_results
        print(Fore.YELLOW + f"{self.results}")
        return self.results

    def run_benchmark(self, model_name, seq_length, tokenizer, process_index, inference_times, barrier):
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        model.eval()

        # Generate random text tokens
        input_ids = tokenizer.encode("Hello " + "A" * (seq_length - 6), return_tensors='pt').cuda()

        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                outputs = model(input_ids)
        total_time = time.time() - start_time
        inference_times[process_index] = total_time / 100
        barrier.wait()


if __name__ == '__main__':
    benchmark = VisionModelBenchmark(model_type='resnet18', batch_sizes=[1, 2], num_processes_list=[1, 2],)
    result = benchmark.start_benchmark()

    benchmark = LanguageModelBenchmark(model_names=['gpt2'], sequence_lengths=[10, 20], num_processes_list=[1, 2])
    results = benchmark.start_benchmark()
