import torch
import torchvision.models as models
import torch.multiprocessing as mp
from pytorch_benchmark import benchmark

class ModelBenchmark:
    def __init__(self, num_processes=5, model_type='alexnet'):
        self.num_processes = num_processes
        self.model_type = model_type
        self.model_dict = {
            'alexnet': (models.alexnet, 224),
            'densenet': (models.densenet161, 224),
            'googlenet': (models.googlenet, 224),
            'mobilenet': (models.mobilenet_v2, 224),
            'resnet18': (models.resnet18, 224),
            'resnext50_32x4d': (models.resnext50_32x4d, 224),
            'vgg16': (models.vgg16, 224),
        }

    def run_benchmark(self, num_runs=100):
        model_func, input_size = self.model_dict[self.model_type]
        model = model_func(pretrained=True).to("cpu")
        sample = torch.randn(8, 3, input_size, input_size)
        result = benchmark(model, sample, num_runs=num_runs)
        return result

    def start_benchmark(self):
        with mp.get_context('spawn').Pool(self.num_processes) as pool:
            results = pool.map(self.run_benchmark, [100] * self.num_processes)

        # Output results
        for idx, result in enumerate(results):
            print(f"Process {idx + 1} benchmark result: {result}")

if __name__ == '__main__':
    benchmark = ModelBenchmark(num_processes=1, model_type='alexnet')
    benchmark.start_benchmark()

    benchmark = ModelBenchmark(num_processes=1, model_type='densenet')
    benchmark.start_benchmark()

    benchmark = ModelBenchmark(num_processes=1, model_type='googlenet')
    benchmark.start_benchmark()

    benchmark = ModelBenchmark(num_processes=1, model_type='mobilenet')
    benchmark.start_benchmark()

    benchmark = ModelBenchmark(num_processes=1, model_type='resnet18')
    benchmark.start_benchmark()

    benchmark = ModelBenchmark(num_processes=1, model_type='resnext50_32x4d')
    benchmark.start_benchmark()

    benchmark = ModelBenchmark(num_processes=1, model_type='vgg16')
    benchmark.start_benchmark()
