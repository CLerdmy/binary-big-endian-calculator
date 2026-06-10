# file size
# split into cpu_count() equal segments
# worker -> mmap, reads segment, computes partial sum/min/max
# main process aggregates results

# REFERENCES:
# https://medium.com/@AlexanderObregon/understanding-pythons-multiprocessing-module-744dba8d4be4
# https://medium.com/@TomeCode/large-file-processing-with-asyncio-and-mmap-in-python-790108188743
# https://habr.com/ru/articles/1001546/
# https://numpy.org/doc/2.4/reference/simd/index.html
# https://medium.com/@npavfan2facts/top-5-python-zero-copy-moves-for-fast-i-o-8545fdf80640


import sys
import os
import mmap
import multiprocessing
import numpy as np


UINT32_BYTES = 4
MAX_SEGMENT_NUMBERS = 64 * 1024 * 1024
MIN_SEGMENT_NUMBERS = 4 * 1024 * 1024


def process_segment(args: tuple) -> tuple[int, int, int]:
    file_path, start_number, end_number = args

    byte_offset = start_number * UINT32_BYTES
    byte_length = (end_number - start_number) * UINT32_BYTES

    alloc_granularity = mmap.ALLOCATIONGRANULARITY
    aligned_offset = (byte_offset // alloc_granularity) * alloc_granularity
    extra_bytes = byte_offset - aligned_offset
    mapped_length = byte_length + extra_bytes

    with open(file_path, "rb") as file:
        with mmap.mmap(file.fileno(), length=mapped_length, access=mmap.ACCESS_READ, offset=aligned_offset) as memory_map:
            n_elements = byte_length // UINT32_BYTES
            arr_be = np.frombuffer(memory_map, dtype=np.dtype(">u4"), count=n_elements, offset=extra_bytes)
            chunk = np.asarray(arr_be, dtype=np.uint32)

            del arr_be
 
    partial_sum = int(chunk.sum(dtype=np.uint64))
    partial_min = int(chunk.min())
    partial_max = int(chunk.max())
    return partial_sum, partial_min, partial_max


def calculate(file_path: str) -> tuple[int, int, int]:
    file_size_bytes = os.path.getsize(file_path)
    # expected: file_size_bytes % UINT32_BYTES != 0

    total_numbers = file_size_bytes // UINT32_BYTES

    num_cpus = os.cpu_count() or 1

    # segments
    num_segments_by_size = max(1, (total_numbers + MAX_SEGMENT_NUMBERS - 1) // MAX_SEGMENT_NUMBERS)
    num_segments = max(num_cpus, num_segments_by_size)
    segment_size = (total_numbers + num_segments - 1) // num_segments

    segments = []
    for i in range(num_segments):
        start = i * segment_size
        if start >= total_numbers:
            break
        end = min(start + segment_size, total_numbers)
        segments.append((file_path, start, end))
    
    # workers = min(num_cpus, num_segments)
    if total_numbers < MIN_SEGMENT_NUMBERS:
        partial_results = [process_segment((file_path, 0, total_numbers))]
    else:
        num_workers = min(num_cpus, len(segments))
        with multiprocessing.Pool(processes=num_workers) as pool:
            partial_results = pool.map(process_segment, segments, chunksize=1)
    
    # aggregate: python int
    total_sum: int = 0
    global_min: int = partial_results[0][1]
    global_max: int = partial_results[0][2]

    for partial_sum, partial_min, partial_max in partial_results:
        total_sum += partial_sum
        if partial_min < global_min:
            global_min = partial_min
        if partial_max > global_max:
            global_max = partial_max
    return total_sum, global_min, global_max


def main():
    # hint
    if len(sys.argv) != 2:
        print("Usage: python3 calc_data2.py <path_to_file>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    total_sum, global_min, global_max = calculate(file_path)

    print(f"sum={total_sum}")
    print(f"min={global_min}")
    print(f"max={global_max}")


if __name__ == "__main__":
    main()