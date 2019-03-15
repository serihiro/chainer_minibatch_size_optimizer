import time
import threading
import pynvml
import sys
import os
import math
import signal
from chainer.training import extension


class MinibatchSizeOptimizer(extension.Extension):
    def __init__(self, strategy, max_minibatch_size=256, **strategy_options):
        self._strategy = strategy
        self._strategy_options = strategy_options
        self._max_minibatch_size = max_minibatch_size
        signal.signal(signal.SIGTERM, self._stop_gpu_profiler)

    def initialize(self, trainer):
        updater = trainer.updater
        optimizer = updater.get_optimizer('main')
        iterator = trainer.updater.get_iterator("main")

        original_model = optimizer.target

        from chainer import backend
        if not isinstance(original_model.device, backend.GpuDevice):
            print("do nothing")
            return

        copied_model = optimizer.target.copy(mode="copy")
        copied_model.to_device(original_model.device)

        optimizer.setup(copied_model)
        # to avoid reporter error
        trainer.reporter.add_observer("main2", copied_model)

        strategy = self._strategy(updater,
                                  initial_minibatch_size=iterator.batch_size,
                                  max_minibatch_size=self._max_minibatch_size,
                                  **self._strategy_options)
        best_minibatch_size = strategy.apply()

        iterator.reset()
        setattr(iterator, "batch_size", best_minibatch_size)
        updater.iteration = 0

        # These codes are copied from trainer.py
        # https://github.com/chainer/chainer/blob/91365f25b94798e1d331c3864d7939e7508e447f/chainer/training/trainer.py#L16-L23
        try:
            _get_time = time.perf_counter
        except AttributeError:
            if os.name == 'nt':
                _get_time = time.clock
            else:
                _get_time = time.time

        trainer._start_at = _get_time()
        trainer._snapshot_elapsed_time = 0.0
        trainer._final_elapsed_time = None

        optimizer.setup(original_model)

    def _stop_gpu_profiler(self, _signal, _frame):
        GpuProfiler.stop()

    def __call__(self, trainer):
        pass


class GpuProfiler:
    _result = {}
    _thread_is_running = threading.Event()
    _stop_event = threading.Event()
    _thread = None
    _lock = threading.Lock()

    @classmethod
    def start(cls, device_index=0):
        result = True
        cls._lock.acquire()
        if not cls._thread_is_running.is_set():
            cls._thread_is_running.set()
            cls._thread = threading.Thread(target=cls._run, args=[device_index])
            cls._thread.start()
        else:
            result = False
        cls._lock.release()

        return result

    @classmethod
    def stop(cls):
        cls._lock.acquire()
        if (cls._thread is not None) and cls._thread.is_alive():
            cls._stop_event.set()
            cls._thread.join()
            cls._thread = None
            cls._stop_event.clear()
            cls._thread_is_running.clear()
        cls._lock.release()

    @classmethod
    def result(cls):
        return cls._result.copy()

    @classmethod
    def reset_result(cls, device_index):
        cls._lock.acquire()
        if device_index in cls._result:
            cls._result[device_index] = []
        cls._lock.release()

    @classmethod
    def _run(cls, device_index):
        if not device_index in cls._result.keys():
            cls._result[device_index] = []

        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        while not cls._stop_event.is_set():
            cls._result[device_index].append(pynvml.nvmlDeviceGetUtilizationRates(device_handle).gpu)
            time.sleep(1.0)


class _BaseStrategy:
    def apply(self):
        raise NotImplementedError()

    @staticmethod
    def _calculate_std_dev(data, mean):
        return (sum(list(map(lambda v: (mean - v) ** 2, data))) / len(data)) ** (1 / 2)


class MaximizeGpuUsage(_BaseStrategy):
    def __init__(self,
                 updater,
                 initial_minibatch_size=10,
                 max_iteration_size=100,
                 max_minibatch_size=100,
                 gpu_usage_max_limit=99.0,
                 update_execution_count=1000,
                 norm=5,
                 max_update_diff=10):
        self._updater = updater
        self._iterator = updater.get_iterator("main")
        # set update execution count
        self._update_execution_count = update_execution_count
        # initialize model parameters
        self._updater.update()

        self._initial_minibatch_size = initial_minibatch_size
        self._max_iteration_size = max_iteration_size
        self._max_minibatch_size = max_minibatch_size
        self._gpu_usage_max_limit = gpu_usage_max_limit
        self._max_update_diff = max_update_diff
        self._norm = norm

    def apply(self):
        pynvml.nvmlInit()

        current_minibatch_size = self._initial_minibatch_size
        last_checked_minibatch_size = sys.maxsize
        current_elapsed_time = 0
        gpu_usage_mean = 0
        iteration_count = 0

        last_update_type = 'up'
        update_diff = 1
        convergence_check_interval = 5

        while True:
            if current_minibatch_size < 0:
                self._print_log(iteration_count, current_minibatch_size, gpu_usage_mean,
                                gpu_usage_stddev, current_elapsed_time)
                break

            if iteration_count >= self._max_iteration_size or \
                    current_minibatch_size >= self._max_minibatch_size:
                self._print_log(iteration_count, current_minibatch_size, gpu_usage_mean,
                                gpu_usage_stddev, current_elapsed_time)
                break

            gpu_usage_mean, gpu_usage_stddev, current_elapsed_time = self._profile(current_minibatch_size)

            if iteration_count % convergence_check_interval == 0:
                print(
                    f'check {last_checked_minibatch_size} : {current_minibatch_size}')
                if abs(last_checked_minibatch_size - current_minibatch_size) <= self._norm:
                    print('convergence!')
                    self._print_log(iteration_count, current_minibatch_size, gpu_usage_mean,
                                    gpu_usage_stddev, current_elapsed_time)
                    break
                else:
                    last_checked_minibatch_size = current_minibatch_size

            self._print_log(iteration_count, current_minibatch_size, gpu_usage_mean,
                            gpu_usage_stddev, current_elapsed_time)

            if gpu_usage_mean >= self._gpu_usage_max_limit:
                if last_update_type == 'down':
                    update_diff += 1
                else:
                    update_diff = 1

                current_minibatch_size -= min(update_diff, self._max_update_diff)
                last_update_type = 'down'
            else:
                if last_update_type == 'up':
                    update_diff += 1
                else:
                    update_diff = 1

                current_minibatch_size += min(update_diff, self._max_update_diff)
                last_update_type = 'up'

            iteration_count += 1

        pynvml.nvmlShutdown()
        return max(0, min(current_minibatch_size, self._max_minibatch_size))

    def _profile(self, minibatch_size, gpu_device_index=0):
        self._iterator.reset()
        setattr(self._iterator, "batch_size", minibatch_size)

        GpuProfiler.start(gpu_device_index)
        start = time.time()
        for _ in range(self._update_execution_count):
            self._updater.update()
        end = time.time()
        GpuProfiler.stop()
        profile_result = GpuProfiler.result()
        GpuProfiler.reset_result(gpu_device_index)

        elapsed_second = end - start
        gpu_usage = [v for v in profile_result[gpu_device_index] if v > 0]
        gpu_usage.sort(reverse=True)
        top_50_gpu_usage = gpu_usage[0:math.floor(len(gpu_usage) / 2)]

        if len(top_50_gpu_usage) > 0:
            mean_gpu_usage = sum(top_50_gpu_usage) / float(len(top_50_gpu_usage))
        else:
            mean_gpu_usage = 0.0

        return mean_gpu_usage, self._calculate_std_dev(top_50_gpu_usage, mean_gpu_usage), elapsed_second

    @staticmethod
    def _print_log(iteration_count, current_minibatch_size, gpu_usage_mean, gpu_usage_stddev, current_elapsed_time):
        print(f'{iteration_count},{current_minibatch_size},{gpu_usage_mean},{gpu_usage_stddev},{current_elapsed_time}')


class MinimizeEpochElapsedTime(_BaseStrategy):
    def __init__(self,
                 updater,
                 initial_minibatch_size=10,
                 max_minibatch_size=100,
                 max_iteration_size=100,
                 update_execution_count=1000,
                 target_improvement_ratio=0.5,
                 max_update_diff=10,
                 norm=0.05
                 ):
        self._updater = updater
        # initialize model parameters
        self._updater.update()

        self._initial_minibatch_size = initial_minibatch_size
        self._max_iteration_size = max_iteration_size
        self._max_minibatch_size = max_minibatch_size
        self._update_execution_count = update_execution_count
        self._target_improvement_ratio = target_improvement_ratio
        self._max_update_diff = max_update_diff
        self._norm = norm

    def apply(self):
        current_minibatch_size = self._initial_minibatch_size
        last_checked_improve_ratio = 100.0
        current_elapsed_time = 0
        iteration_count = 0

        last_update_type = 'up'
        update_diff = 1
        convergence_check_interval = 5

        while True:
            if current_minibatch_size < 0:
                self._print_log(iteration_count, current_minibatch_size, current_improve_ratio, current_elapsed_time)
                break

            if iteration_count >= self._max_iteration_size or \
                    current_minibatch_size >= self._max_minibatch_size:
                self._print_log(iteration_count, current_minibatch_size, current_improve_ratio, current_elapsed_time)
                break

            current_elapsed_time = self._profile(current_minibatch_size)

            if iteration_count == 0:
                base_elapsed_time = current_elapsed_time
            current_improve_ratio = (base_elapsed_time - current_elapsed_time) / float(base_elapsed_time)

            if iteration_count % convergence_check_interval == 0:
                print(
                    f'check {last_checked_improve_ratio} : {current_improve_ratio}')
                if abs(last_checked_improve_ratio - current_improve_ratio) <= self._norm:
                    print('convergence!')
                    self._print_log(iteration_count, current_minibatch_size, current_improve_ratio,
                                    current_elapsed_time)
                    break
                else:
                    last_checked_improve_ratio = current_improve_ratio

            self._print_log(iteration_count, current_minibatch_size, current_improve_ratio, current_elapsed_time)
            if (current_improve_ratio >= 0) and (current_improve_ratio < self._target_improvement_ratio):
                if last_update_type == 'up':
                    update_diff += 1
                else:
                    update_diff = 2

                current_minibatch_size += min(update_diff, self._max_update_diff)
                last_update_type = 'up'
            else:
                if last_update_type == 'down':
                    update_diff += 1
                else:
                    update_diff = 2

                current_minibatch_size -= min(update_diff, self._max_update_diff)
                last_update_type = 'down'

            iteration_count += 1

        return max(0, min(current_minibatch_size, self._max_minibatch_size))

    def _profile(self, minibatch_size):
        iterator = self._updater.get_iterator("main")
        iterator.reset()
        setattr(iterator, "batch_size", minibatch_size)
        data_size = len(iterator.dataset)

        start = time.time()
        for i in range(self._update_execution_count):
            self._updater.update()
        end = time.time()

        elapsed_time = end - start
        estimated_1_epoch_time = (elapsed_time / float(self._update_execution_count)) * math.ceil(
            data_size / float(minibatch_size))

        return estimated_1_epoch_time

    @staticmethod
    def _print_log(iteration_count, current_minibatch_size, current_improve_ratio, current_elapsed_time):
        print(f'{iteration_count},{current_minibatch_size},{current_improve_ratio},{current_elapsed_time}')
