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
from BenchmarkMaker import LanguageModelBenchmark

mp.set_start_method('spawn', force=True)


model_types = ['bert', 'distilgpt2', 'gpt2', 'RoBERTa', 'T5']
sequence_lengths = [1]
num_processes_list = [1]


@pytest.mark.parametrize("model_type", model_types)
@pytest.mark.parametrize("seq_length", sequence_lengths)
@pytest.mark.parametrize("num_processes", num_processes_list)
def test_language_model_benchmark(model_type, seq_length, num_processes):
    """Test that each language model runs benchmarks correctly across different sequence lengths and number of processes."""
    benchmark = LanguageModelBenchmark(
        model_types=[model_type],
        sequence_lengths=[seq_length],
        num_processes_list=[num_processes],
        device='cpu'
    )
    # Run Test
    benchmark.start_benchmark()
    # Validation Result
    assert model_type in benchmark.results
    # Validation Process
    assert f'processes_{num_processes}' in benchmark.results[model_type][f'seq_len_{seq_length}']
    # Validation Time
    assert 'average_inference_time' in benchmark.results[model_type][f'seq_len_{seq_length}'][f'processes_{num_processes}']
    assert benchmark.results[model_type][f'seq_len_{seq_length}'][f'processes_{num_processes}']['average_inference_time'] > 0


@pytest.mark.parametrize("model_type", model_types)
@pytest.mark.parametrize("seq_length", sequence_lengths)
@pytest.mark.parametrize("num_processes", num_processes_list)
def test_save_results_to_csv(model_type, seq_length, num_processes, tmp_path):
    """Test saving results to CSV file for each language model configuration."""
    benchmark = LanguageModelBenchmark(
        model_types=[model_type],
        sequence_lengths=[seq_length],
        num_processes_list=[num_processes],
        device='cpu'
    )
    # Run benchmark to populate results
    benchmark.start_benchmark()

    # Define a temporary file path
    temp_file = tmp_path / "test_results.csv"
    # Save results to the temporary file
    benchmark.save_results_to_csv(temp_file)

    # Read the saved file and check its content
    with open(temp_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        expected_headers = [
            'Model Type', 'Sequence Length', 'Number of Processes', 'Overall Time (s)', 'Average Inference Time (s)']
        assert headers == expected_headers, "CSV headers do not match expected headers"

        found = False
        for row in reader:
            if row[0] == model_type and row[1] == str(seq_length) and row[2] == str(num_processes):
                found = True
                # Check if the times are recorded and positive
                assert float(row[3]) > 0, "Overall Time should be greater than 0"
                assert float(row[4]) > 0, "Average Inference Time should be greater than 0"
        assert found, "Expected results not found in the CSV file"
