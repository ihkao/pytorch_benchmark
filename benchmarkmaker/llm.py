import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorch_benchmark import benchmark

class LanguageModelBenchmark:
    def __init__(self, num_processes=5, model_name='gpt2'):
        self.num_processes = num_processes
        self.model_name = model_name
        self.model_dict = {
            'bert': 'bert-base-uncased',
            'gpt2': 'gpt2',
            'roberta': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            't5-small': 't5-small',
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dict[self.model_name])
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dict[self.model_name]).to("cpu")

    def run_benchmark(self, num_runs=100):
        # Prepare input for the model
        input_ids = self.tokenizer.encode("Hello, world!", return_tensors="pt").to("cpu")
        result = benchmark(self.model, input_ids, num_runs=num_runs)
        return result

    def start_benchmark(self):
        with mp.get_context('spawn').Pool(self.num_processes) as pool:
            results = pool.map(self.run_benchmark, [100] * self.num_processes)

        # Output results
        for idx, result in enumerate(results):
            print(f"Process {idx + 1} benchmark result: {result}")

if __name__ == '__main__':
    # benchmark = LanguageModelBenchmark(num_processes=1, model_name='gpt2')
    # benchmark.start_benchmark()

    benchmark = LanguageModelBenchmark(num_processes=1, model_name='bert')
    benchmark.start_benchmark()

    # benchmark = LanguageModelBenchmark(num_processes=1, model_name='roberta')
    # benchmark.start_benchmark()

    # benchmark = LanguageModelBenchmark(num_processes=1, model_name='distilbert')
    # benchmark.start_benchmark()

    # benchmark = LanguageModelBenchmark(num_processes=1, model_name='t5-small')
    # benchmark.start_benchmark()
