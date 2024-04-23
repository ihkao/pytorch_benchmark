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
import pytest
import torch.multiprocessing as mp
from BenchmarkMaker import VisionModelBenchmark

mp.set_start_method('spawn', force=True)


model_types = ['alexnet', 'densenet', 'googlenet', 'mobilenet', 'resnet18', 'resnext50_32x4d', 'vgg16']
batch_sizes = [1]
num_processes_list = [1]


@pytest.mark.parametrize("model_type", model_types)
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("num_processes", num_processes_list)
def test_model_benchmark(model_type, batch_size, num_processes, tmp_path):
    """Test that each model runs benchmarks correctly across different batch sizes and number of processes."""
    benchmark = VisionModelBenchmark(
        model_types=[model_type],
        batch_sizes=[batch_size],
        num_processes_list=[num_processes],
        device='cuda'
    )
    # Run Test
    benchmark.start_benchmark()
    # Validation Result
    assert model_type in benchmark.results
    # Validation Process
    assert f'processes_{num_processes}' in benchmark.results[model_type][f'batchsize_{batch_size}']
    # Validation Time
    assert 'average_inference_time' in benchmark.results[model_type][f'batchsize_{batch_size}'][f'processes_{num_processes}']
    assert benchmark.results[model_type][f'batchsize_{batch_size}'][f'processes_{num_processes}']['average_inference_time'] > 0

    # Define a temporary file path
    temp_file = tmp_path / "test_results.csv"
    # Save results to the temporary file
    benchmark.save_results_to_csv(temp_file)

    # Read the saved file and check its content
    with open(temp_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        expected_headers = [
            'Model Type', 'Batch Size', 'Number of Processes', 'Overall Time (s)', 'Average Inference Time (s)']
        assert headers == expected_headers, "CSV headers do not match expected headers"

        found = False
        for row in reader:
            if row[0] == model_type and row[1] == str(batch_size) and row[2] == str(num_processes):
                found = True
                # Check if the times are recorded and positive
                assert float(row[3]) > 0, "Overall Time should be greater than 0"
                assert float(row[4]) > 0, "Average Inference Time should be greater than 0"
        assert found, "Expected results not found in the CSV file"   
