# Use mobilint-cli

English | [한국어](usage.KR.md)

This guide covers common tasks: checking NPU status, inspecting an MXQ file, profiling inference, and measuring throughput.

## Download YOLO11m for ARIES-RB

These examples use CLI v1.6.0 and [Mobilint's YOLO11m](https://huggingface.co/mobilint/YOLO11m). No compilation is required.
From the repository root, download the latest [ARIES-RB MXQ](https://huggingface.co/mobilint/YOLO11m/blob/main/aries-rb/yolo11m.mxq).

```bash
curl -fL -o yolo11m.mxq \
  https://huggingface.co/mobilint/YOLO11m/resolve/main/aries-rb/yolo11m.mxq
```

Run the remaining examples from the same directory.

## Check NPU status

```bash
mobilint-cli devices
mobilint-cli status
```

- `devices` lists the device number and model name.
- `status` reports temperature, memory use, and utilization for a Mobilint NPU, much like `nvidia-smi` does for an NVIDIA GPU.
- To refresh once per second, run `mobilint-cli status -l 1`. Press `Ctrl+C` to stop.

![Status of the ARIES-RB at device 0](../assets/devices-status.png)

## mxqtool: Inspect an MXQ file

```bash
mobilint-cli mxqtool show yolo11m.mxq
```

`show` reads the file without running inference. In the sample output:

- `Compiler Version`: The version of the compiler that built the file, not the installed CLI version.
- `Target Hardware`: The device the model was compiled for (`aries-rb`).
- `Input Section #0`: The first input accepts `Float32`, `Float16`, or `Int8` user data. The NPU uses `Int8`.

![MXQ header and first input section](../assets/mxqtool.png)

## summary: Check model requirements

```bash
mobilint-cli summary yolo11m.mxq
```

`summary` lists the target device, supported core modes, and model inputs and outputs.

- Target: `aries-rb`. The model supports `Single`, `Multi`, `Global4`, and `Global8`. Both examples below use `global8`.
- Input: One `Float32` tensor of shape `(640, 640, 3)`.
- Output: Six `Float32` tensors.

![YOLO11m target, supported core modes, and I/O](../assets/summary.png)

## testinfer: Profile inference

Record one inference with `testinfer`, then open the trace in Perfetto to inspect the time spent in each stage.

```bash
mobilint-cli testinfer --core-mode global8 --input-mode seq --repeat-count 1 \
  --trace-path yolo11m-trace.json yolo11m.mxq
```

- `--input-mode seq` generates sequential input values instead of loading a photo.
- `--repeat-count 1` runs one inference.
- `average duration` reports execution time in nanoseconds (ns). It does not measure detection accuracy.

`--trace-path` writes stage events to a JSON file. The command above saves `yolo11m-trace.json` in the current directory.

The screenshot uses `utilities/sample/yolo11m-trace.json` so you can open the same trace as the example.

![Test inference with one input, six outputs, and the elapsed time](../assets/testinfer.png)

### Read stage durations in Perfetto

1. In the [Perfetto UI](https://ui.perfetto.dev/), select `Open trace file`.
2. Open the JSON file you just created. Without an NPU, you can download and open the [sample trace](../sample/yolo11m-trace.json) instead.
3. Select an interval on the timeline. Read its `Duration` under `Current Selection`.

In the sample, `create model` took 10.888 ms, `launch` took 17.143 ms, and `infer` took 6.328 ms.

The selected `run npu` interval took 3.251 ms. It overlaps `infer` on another thread. Do not add the two durations.

![Perfetto selection showing a run npu duration of 3 ms 251 µs](../assets/perfetto.png)

## benchmark: Measure inference throughput

```bash
mobilint-cli benchmark -p yolo11m.mxq -i float --core-mode global8 --thread 1 -t 10
```

- `-i float` uses floating-point input, which this model supports.
- `--thread 1` uses one thread.
- `-t 10` runs inference repeatedly for 10 seconds.

Read the two result fields separately:

- `Total Inference` counts completed inferences.
- `Avg Inference` reports the average number per second, not the time per inference.

The sample run completed 2,147 inferences at an average of 214.62 per second. Use the same model, core mode, and thread count when comparing results.

![Benchmark settings and results for a 10-second run](../assets/benchmark.png)

For more commands and options, see the [official CLI usage guide](https://docs.mobilint.com/runtime/en/v1.6/utility_usage/#mobilint-cli).
